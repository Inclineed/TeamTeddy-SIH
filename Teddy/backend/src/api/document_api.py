"""
FastAPI integration for document processing and chunking
Provides REST API endpoints for uploading and processing documents
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.document_processor import DocumentProcessor, ChunkingConfig, ProcessedDocument

# Initialize APIRouter (this can be imported into main server.py)
doc_app = APIRouter()

# Configuration models
class ChunkingConfigModel(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"

class ChunkMetadata(BaseModel):
    chunk_index: int
    total_chunks: int
    chunk_size: int
    splitter_type: str
    source_file: str
    file_type: str
    filename: str

class DocumentChunk(BaseModel):
    content: str
    metadata: ChunkMetadata

class ProcessedDocumentResponse(BaseModel):
    filename: str
    file_type: str
    total_chunks: int
    chunks_preview: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_status: str

class DocumentListResponse(BaseModel):
    documents: List[ProcessedDocumentResponse]
    total_documents: int

# Global processor instance
processor = DocumentProcessor()

# Use the same directory that search API expects (correct location)
UPLOAD_DIR = Path(__file__).parent / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@doc_app.post("/upload", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a single document for processing
    """
    result = await _upload_single_file(file)
    return result

@doc_app.post("/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents for processing
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    errors = []
    
    for file in files:
        try:
            result = await _upload_single_file(file)
            results.append(result)
        except HTTPException as e:
            errors.append({
                "filename": file.filename or "unknown",
                "error": e.detail
            })
        except Exception as e:
            errors.append({
                "filename": file.filename or "unknown", 
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful_uploads": len(results),
        "failed_uploads": len(errors),
        "results": results,
        "errors": errors,
        "message": f"Uploaded {len(results)}/{len(files)} files successfully"
    }

async def _upload_single_file(file: UploadFile) -> Dict[str, str]:
    """
    Helper function to upload a single file
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf']
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(supported_extensions)}"
        )
    
    # Handle duplicate filenames by adding timestamp
    original_filename = file.filename
    file_path = UPLOAD_DIR / original_filename
    
    if file_path.exists():
        # Add timestamp to avoid overwriting
        timestamp = int(time.time())
        name_part = Path(original_filename).stem
        extension = Path(original_filename).suffix
        new_filename = f"{name_part}_{timestamp}{extension}"
        file_path = UPLOAD_DIR / new_filename
        actual_filename = new_filename
    else:
        actual_filename = original_filename
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "original_filename": original_filename,
            "saved_filename": actual_filename,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "status": "uploaded",
            "message": f"File {actual_filename} uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file {original_filename}: {str(e)}")

@doc_app.post("/process", response_model=ProcessedDocumentResponse)
async def process_document(
    filename: str,
    chunk_size: int = Query(1000, description="Size of each chunk"),
    chunk_overlap: int = Query(200, description="Overlap between chunks"),
    splitter_type: str = Query("recursive", description="Type of text splitter")
):
    """
    Process an uploaded document and return chunks
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        # Configure chunking
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Update processor config
        global processor
        processor = DocumentProcessor(config)
        
        # Process the document
        processed_doc = processor.process_file(str(file_path), splitter_type)
        
        # Get chunk previews
        previews = processor.get_chunk_preview(processed_doc.chunks, 3)
        
        # Filter out non-serializable items from metadata
        clean_metadata = {k: v for k, v in processed_doc.metadata.items() 
                         if not callable(v) and not isinstance(v, type)}
        
        return ProcessedDocumentResponse(
            filename=processed_doc.filename,
            file_type=processed_doc.file_type,
            total_chunks=processed_doc.total_chunks,
            chunks_preview=previews,
            metadata=clean_metadata,
            processing_status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@doc_app.post("/upload-and-process")
async def upload_and_process_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(1000, description="Size of each chunk"),
    chunk_overlap: int = Query(200, description="Overlap between chunks"),
    splitter_type: str = Query("recursive", description="Type of text splitter"),
    auto_index: bool = Query(True, description="Automatically index for search after processing")
):
    """
    Upload and immediately process a document in one step
    """
    try:
        # Step 1: Upload the file
        upload_result = await _upload_single_file(file)
        filename = upload_result["saved_filename"]
        
        # Step 2: Process the document
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"
        )
        
        global processor
        processor = DocumentProcessor(config)
        file_path = UPLOAD_DIR / filename
        processed_doc = processor.process_file(str(file_path), splitter_type)
        
        # Step 3: Auto-index for search if requested
        index_result = None
        if auto_index:
            try:
                # Direct indexing without HTTP call to avoid circular dependency
                from src.core.semantic_search import SemanticSearchEngine
                
                search_engine = SemanticSearchEngine()
                indexed_chunks = search_engine.add_documents(processed_doc.chunks)
                
                index_result = {
                    "indexed_chunks": indexed_chunks,
                    "filename": filename,
                    "status": "success"
                }
            except Exception as e:
                index_result = {"error": f"Indexing failed: {str(e)}"}
        
        # Get chunk previews
        previews = processor.get_chunk_preview(processed_doc.chunks, 3)
        
        # Clean metadata to avoid serialization issues
        clean_metadata = {k: v for k, v in processed_doc.metadata.items() 
                         if not callable(v) and not isinstance(v, type)}
        
        return {
            "upload": upload_result,
            "processing": {
                "filename": processed_doc.filename,
                "file_type": processed_doc.file_type,
                "total_chunks": processed_doc.total_chunks,
                "chunks_preview": previews,
                "metadata": clean_metadata,
                "processing_status": "success"
            },
            "indexing": index_result,
            "status": "completed",
            "message": f"File {filename} uploaded, processed, and {'indexed' if index_result and 'error' not in index_result else 'processing failed'} successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload and process document: {str(e)}")

@doc_app.post("/upload-and-process-multiple")
async def upload_and_process_multiple_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(1000, description="Size of each chunk"),
    chunk_overlap: int = Query(200, description="Overlap between chunks"),
    splitter_type: str = Query("recursive", description="Type of text splitter"),
    auto_index: bool = Query(True, description="Automatically index for search after processing"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process multiple documents in one step
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Process each file individually
            result = await upload_and_process_document(
                file=file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type=splitter_type,
                auto_index=auto_index
            )
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            errors.append({
                "filename": file.filename or "unknown",
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful_processes": len(results),
        "failed_processes": len(errors),
        "results": results,
        "errors": errors,
        "message": f"Processed {len(results)}/{len(files)} files successfully"
    }

@doc_app.get("/chunks/{filename}", response_model=List[DocumentChunk])
async def get_document_chunks(
    filename: str,
    start_index: int = Query(0, description="Starting chunk index"),
    limit: int = Query(10, description="Number of chunks to return")
):
    """
    Get specific chunks from a processed document
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        # Process the document (could be cached in production)
        processed_doc = processor.process_file(str(file_path))
        
        # Get requested chunks
        end_index = min(start_index + limit, len(processed_doc.chunks))
        requested_chunks = processed_doc.chunks[start_index:end_index]
        
        # Convert to response format
        response_chunks = []
        for chunk in requested_chunks:
            chunk_metadata = ChunkMetadata(**chunk.metadata)
            response_chunks.append(
                DocumentChunk(
                    content=chunk.page_content,
                    metadata=chunk_metadata
                )
            )
        
        return response_chunks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")

@doc_app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents and their processing status
    """
    try:
        documents = []
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt']:
                try:
                    # Get basic file info
                    processed_doc = processor.process_file(str(file_path))
                    previews = processor.get_chunk_preview(processed_doc.chunks, 2)
                    
                    # Filter out non-serializable items from metadata
                    clean_metadata = {k: v for k, v in processed_doc.metadata.items() 
                                    if not callable(v) and not isinstance(v, type)}
                    
                    documents.append(ProcessedDocumentResponse(
                        filename=processed_doc.filename,
                        file_type=processed_doc.file_type,
                        total_chunks=processed_doc.total_chunks,
                        chunks_preview=previews,
                        metadata=clean_metadata,
                        processing_status="processed"
                    ))
                except Exception as e:
                    # If processing fails, still show the file
                    documents.append(ProcessedDocumentResponse(
                        filename=file_path.name,
                        file_type=file_path.suffix.lower(),
                        total_chunks=0,
                        chunks_preview=[],
                        metadata={"error": str(e)},
                        processing_status="error"
                    ))
        
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@doc_app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete an uploaded document
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@doc_app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "document-processing-api",
        "supported_formats": [".pdf", ".docx", ".doc", ".txt"]
    }

@doc_app.get("/export/{filename}")
async def export_chunks(filename: str):
    """
    Export document chunks to a text file
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        processed_doc = processor.process_file(str(file_path))
        export_path = processor.export_chunks_to_text(processed_doc, str(UPLOAD_DIR))
        
        return {
            "filename": filename,
            "export_path": export_path,
            "total_chunks": processed_doc.total_chunks,
            "message": "Chunks exported successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export chunks: {str(e)}")

# Example usage and testing function
async def test_processing():
    """
    Test function to demonstrate the processing capabilities
    """
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if test_files:
        test_file = test_files[0]
        print(f"Testing with file: {test_file}")
        
        try:
            processed_doc = processor.process_file(str(test_file))
            print(f"Successfully processed: {processed_doc.filename}")
            print(f"Total chunks: {processed_doc.total_chunks}")
            
            # Show first chunk
            if processed_doc.chunks:
                first_chunk = processed_doc.chunks[0]
                print(f"First chunk preview: {first_chunk.page_content[:200]}...")
                print(f"Chunk metadata: {first_chunk.metadata}")
                
        except Exception as e:
            print(f"Error during testing: {e}")
    else:
        print("No test files found in data directory")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_processing())