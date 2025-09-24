"""
FastAPI integration for semantic search functionality
Provides REST API endpoints for semantic search operations
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from src.core.semantic_search import SemanticSearchEngine, SearchResult, EmbeddingConfig
from src.core.document_processor import DocumentProcessor, ChunkingConfig

# Initialize APIRouter
search_app = APIRouter()

# Configuration models
class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: Optional[float] = None

class IndexDocumentRequest(BaseModel):
    filename: str
    chunk_size: int = 800
    chunk_overlap: int = 150
    splitter_type: str = "recursive"

class CollectionStats(BaseModel):
    total_chunks: int
    collection_name: str
    sample_source_files: List[str]
    persist_directory: str

# Global search engine instance
search_engine = None

def get_search_engine():
    """Get or create the search engine instance"""
    global search_engine
    if search_engine is None:
        search_engine = SemanticSearchEngine(
            collection_name="document_chunks",
            embedding_config=EmbeddingConfig(batch_size=16)
        )
    return search_engine

@search_app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    try:
        get_search_engine()
        print("✅ Semantic search engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")

@search_app.post("/search", response_model=SearchResponse)
async def semantic_search(search_query: SearchQuery):
    """
    Perform semantic search on indexed documents
    """
    try:
        start_time = time.time()
        
        engine = get_search_engine()
        results = engine.search(
            query=search_query.query,
            n_results=search_query.n_results,
            filter_metadata=search_query.filter_metadata
        )
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Convert SearchResult objects to dictionaries
        result_dicts = []
        for result in results:
            result_dict = {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "source_file": result.source_file,
                "chunk_index": result.chunk_index
            }
            result_dicts.append(result_dict)
        
        return SearchResponse(
            query=search_query.query,
            results=result_dicts,
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@search_app.get("/search", response_model=SearchResponse)
async def semantic_search_get(
    q: str = Query(..., description="Search query"),
    n_results: int = Query(5, description="Number of results to return"),
    source_file: Optional[str] = Query(None, description="Filter by source file")
):
    """
    Perform semantic search via GET request
    """
    filter_metadata = None
    if source_file:
        filter_metadata = {"source_file": source_file}
    
    search_query = SearchQuery(
        query=q,
        n_results=n_results,
        filter_metadata=filter_metadata
    )
    
    return await semantic_search(search_query)

@search_app.post("/index")
async def index_document(
    request: IndexDocumentRequest,
    background_tasks: BackgroundTasks
):
    """
    Index a document for semantic search
    """
    try:
        # Check if file exists
        data_dir = Path(__file__).parent.parent.parent / "data"
        file_path = data_dir / request.filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        
        # Add indexing task to background
        background_tasks.add_task(
            index_document_background,
            str(file_path),
            request.chunk_size,
            request.chunk_overlap,
            request.splitter_type
        )
        
        return {
            "message": f"Document {request.filename} queued for indexing",
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue document for indexing: {str(e)}")

async def index_document_background(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str
):
    """
    Background task to index a document
    """
    try:
        # Initialize processor
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        processor = DocumentProcessor(config)
        
        # Process document
        processed_doc = processor.process_file(file_path, splitter_type)
        
        # Add to search engine
        engine = get_search_engine()
        engine.add_documents(processed_doc)
        
        print(f"✅ Successfully indexed {processed_doc.filename}")
        
    except Exception as e:
        print(f"❌ Failed to index document {file_path}: {e}")

@search_app.post("/index-all")
async def index_all_documents(background_tasks: BackgroundTasks):
    """
    Index all documents in the data directory
    """
    try:
        data_dir = Path(__file__).parent.parent.parent / "data"
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
        
        files_to_index = []
        for ext in supported_extensions:
            files_to_index.extend(list(data_dir.glob(f"*{ext}")))
        
        if not files_to_index:
            return {
                "message": "No supported documents found in data directory",
                "files_found": 0
            }
        
        # Add indexing tasks to background
        for file_path in files_to_index:
            background_tasks.add_task(
                index_document_background,
                str(file_path),
                800,  # default chunk_size
                150,  # default chunk_overlap
                "recursive"  # default splitter_type
            )
        
        return {
            "message": f"Queued {len(files_to_index)} documents for indexing",
            "files": [f.name for f in files_to_index],
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue documents for indexing: {str(e)}")

@search_app.get("/stats", response_model=CollectionStats)
async def get_collection_stats():
    """
    Get statistics about the indexed document collection
    """
    try:
        engine = get_search_engine()
        stats = engine.get_collection_stats()
        
        return CollectionStats(
            total_chunks=stats.get("total_chunks", 0),
            collection_name=stats.get("collection_name", ""),
            sample_source_files=stats.get("sample_source_files", []),
            persist_directory=stats.get("persist_directory", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {str(e)}")

@search_app.delete("/collection")
async def clear_collection():
    """
    Clear all documents from the collection
    """
    try:
        engine = get_search_engine()
        engine.clear_collection()
        
        return {
            "message": "Collection cleared successfully",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

@search_app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Test embedding generation
        from src.core.semantic_search import OllamaEmbeddingGenerator
        embedding_gen = OllamaEmbeddingGenerator()
        test_embedding = embedding_gen.generate_embedding("test")
        
        # Test ChromaDB connection
        engine = get_search_engine()
        stats = engine.get_collection_stats()
        
        return {
            "status": "healthy",
            "service": "semantic-search-api",
            "embedding_model": "bge-m3",
            "embedding_dimensions": len(test_embedding),
            "indexed_chunks": stats.get("total_chunks", 0),
            "vector_database": "ChromaDB"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "semantic-search-api"
        }

@search_app.get("/similar/{chunk_id}")
async def find_similar_chunks(
    chunk_id: str,
    n_results: int = Query(5, description="Number of similar chunks to return")
):
    """
    Find chunks similar to a specific chunk
    """
    try:
        engine = get_search_engine()
        
        # Get the chunk content first
        chunk_data = engine.collection.get(ids=[chunk_id], include=['documents'])
        
        if not chunk_data['documents'] or not chunk_data['documents'][0]:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        chunk_content = chunk_data['documents'][0]
        
        # Use the chunk content as query to find similar chunks
        results = engine.search(
            query=chunk_content,
            n_results=n_results + 1  # +1 because the original chunk will be included
        )
        
        # Filter out the original chunk
        similar_results = [r for r in results if r.chunk_id != chunk_id][:n_results]
        
        # Convert to response format
        result_dicts = []
        for result in similar_results:
            result_dict = {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "source_file": result.source_file,
                "chunk_index": result.chunk_index
            }
            result_dicts.append(result_dict)
        
        return {
            "original_chunk_id": chunk_id,
            "similar_chunks": result_dicts,
            "total_similar": len(similar_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar chunks: {str(e)}")
