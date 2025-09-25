"""
Document Processing Module using LangChain
This module provides functionality to parse PDF/DOC files and convert them into chunks
ready for embedding generation.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredPDFLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from langchain.schema import Document

# Import audio processor for STT functionality
from .audio_processor import create_audio_processor, AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    keep_separator: bool = True
    length_function: callable = len
    is_separator_regex: bool = False

@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    filename: str
    file_path: str
    file_type: str
    total_chunks: int
    chunks: List[Document]
    metadata: Dict[str, Any]

class DocumentProcessor:
    """
    Main class for processing documents and converting them into chunks
    """
    
    def __init__(self, config: ChunkingConfig = None, enable_audio: bool = True):
        """
        Initialize the document processor
        
        Args:
            config: ChunkingConfig object with chunking parameters
            enable_audio: Whether to enable audio file processing (STT)
        """
        self.config = config or ChunkingConfig()
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_doc,
            '.txt': self._load_txt
        }
        
        # Add audio support if enabled
        self.enable_audio = enable_audio
        if enable_audio:
            try:
                self.audio_processor = create_audio_processor(backend="whisper")
                # Add audio extensions
                audio_extensions = {
                    '.mp3': self._load_audio,
                    '.wav': self._load_audio,
                    '.m4a': self._load_audio,
                    '.flac': self._load_audio,
                    '.aac': self._load_audio,
                    '.ogg': self._load_audio,
                    '.wma': self._load_audio,
                    '.mp4': self._load_audio,  # Audio from video
                    '.avi': self._load_audio,  # Audio from video
                    '.mov': self._load_audio,  # Audio from video
                    '.mkv': self._load_audio   # Audio from video
                }
                self.supported_extensions.update(audio_extensions)
                logger.info("Audio processing enabled with Whisper STT")
            except Exception as e:
                logger.warning(f"Failed to initialize audio processor: {e}")
                self.enable_audio = False
                self.audio_processor = None
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        return Path(file_path).suffix.lower()
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document using PyPDFLoader"""
        try:
            # Try PyPDFLoader first (faster, structured)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded PDF with PyPDFLoader: {file_path}")
            return documents
        except Exception as e:
            logger.warning(f"PyPDFLoader failed for {file_path}, trying UnstructuredPDFLoader: {e}")
            try:
                # Fallback to UnstructuredPDFLoader (more robust)
                loader = UnstructuredPDFLoader(file_path)
                documents = loader.load()
                logger.info(f"Successfully loaded PDF with UnstructuredPDFLoader: {file_path}")
                return documents
            except Exception as e2:
                logger.error(f"Failed to load PDF {file_path}: {e2}")
                raise
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX document"""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded DOCX: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise
    
    def _load_doc(self, file_path: str) -> List[Document]:
        """Load DOC document"""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded DOC: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load DOC {file_path}: {e}")
            raise
    
    def _load_txt(self, file_path: str) -> List[Document]:
        """Load TXT document"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Successfully loaded TXT: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            raise
    
    def _load_audio(self, file_path: str) -> List[Document]:
        """Load audio document using Speech-to-Text"""
        if not self.enable_audio or not self.audio_processor:
            raise ValueError("Audio processing is not enabled")
        
        try:
            filename = Path(file_path).name
            logger.info(f"Starting audio transcription: {filename}")
            
            # Process audio and get transcription
            audio_result = self.audio_processor.process_audio_for_rag(file_path, filename)
            
            # Create Document object similar to text documents
            document = Document(
                page_content=audio_result["content"],
                metadata={
                    "source": file_path,
                    "filename": filename,
                    **audio_result["metadata"]
                }
            )
            
            logger.info(f"Successfully transcribed audio: {filename} ({len(audio_result['content'])} characters)")
            return [document]
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file type
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self._get_file_type(file_path)
        if file_type not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        loader_func = self.supported_extensions[file_type]
        documents = loader_func(file_path)
        
        # Add file metadata to documents
        for doc in documents:
            doc.metadata.update({
                'source_file': file_path,
                'file_type': file_type,
                'filename': Path(file_path).name
            })
        
        return documents
    
    def create_text_splitter(self, splitter_type: str = "recursive") -> Any:
        """
        Create a text splitter based on the specified type
        
        Args:
            splitter_type: Type of splitter ("recursive", "character", "token")
            
        Returns:
            Text splitter instance
        """
        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function,
                separators=["\n\n", "\n", " ", ""]
            )
        
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.separator,
                length_function=self.config.length_function
            )
        
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    def chunk_documents(self, documents: List[Document], splitter_type: str = "recursive") -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects to chunk
            splitter_type: Type of text splitter to use
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = self.create_text_splitter(splitter_type)
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk.page_content),
                'splitter_type': splitter_type
            })
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def process_file(self, file_path: str, splitter_type: str = "recursive") -> ProcessedDocument:
        """
        Complete pipeline to process a single file
        
        Args:
            file_path: Path to the file to process
            splitter_type: Type of text splitter to use
            
        Returns:
            ProcessedDocument object containing all processing results
        """
        # Load the document
        documents = self.load_document(file_path)
        
        # Chunk the documents
        chunks = self.chunk_documents(documents, splitter_type)
        
        # Create processed document object
        processed_doc = ProcessedDocument(
            filename=Path(file_path).name,
            file_path=file_path,
            file_type=self._get_file_type(file_path),
            total_chunks=len(chunks),
            chunks=chunks,
            metadata={
                'original_documents': len(documents),
                'chunking_config': self.config.__dict__,
                'splitter_type': splitter_type
            }
        )
        
        logger.info(f"Processed file: {file_path} -> {len(chunks)} chunks")
        return processed_doc
    
    def process_directory(self, directory_path: str, splitter_type: str = "recursive") -> List[ProcessedDocument]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            splitter_type: Type of text splitter to use
            
        Returns:
            List of ProcessedDocument objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        processed_docs = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    processed_doc = self.process_file(str(file_path), splitter_type)
                    processed_docs.append(processed_doc)
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {len(processed_docs)} files from directory: {directory_path}")
        return processed_docs
    
    def _clean_for_serialization(self, obj: Any) -> Any:
        """
        Recursively clean an object to make it JSON serializable
        """
        if callable(obj):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._clean_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_serialization(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, convert to dict but filter out methods
            try:
                obj_dict = vars(obj)
                return {k: self._clean_for_serialization(v) for k, v in obj_dict.items() 
                       if not k.startswith('_') and not callable(v)}
            except:
                return str(obj)
        else:
            # For basic types (int, str, float, bool, None), return as-is
            try:
                import json
                json.dumps(obj)  # Test if it's serializable
                return obj
            except:
                return str(obj)

    def get_chunk_preview(self, chunks: List[Document], num_chunks: int = 3) -> List[Dict[str, Any]]:
        """
        Get a preview of the first few chunks for inspection
        
        Args:
            chunks: List of Document chunks
            num_chunks: Number of chunks to preview
            
        Returns:
            List of dictionaries containing chunk previews
        """
        previews = []
        for i, chunk in enumerate(chunks[:num_chunks]):
            # Use comprehensive cleaning for metadata
            clean_metadata = self._clean_for_serialization(chunk.metadata)
            
            preview = {
                'chunk_index': i,
                'content_preview': chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                'content_length': len(chunk.page_content),
                'metadata': clean_metadata
            }
            previews.append(preview)
        
        return previews
    
    def export_chunks_to_text(self, processed_doc: ProcessedDocument, output_dir: str = None) -> str:
        """
        Export chunks to a text file for inspection
        
        Args:
            processed_doc: ProcessedDocument object
            output_dir: Directory to save the output file
            
        Returns:
            Path to the exported file
        """
        if output_dir is None:
            output_dir = Path(processed_doc.file_path).parent
        
        output_path = Path(output_dir) / f"{processed_doc.filename}_chunks.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Document: {processed_doc.filename}\n")
            f.write(f"Total chunks: {processed_doc.total_chunks}\n")
            f.write(f"File type: {processed_doc.file_type}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, chunk in enumerate(processed_doc.chunks):
                f.write(f"CHUNK {i + 1}/{processed_doc.total_chunks}\n")
                f.write(f"Length: {len(chunk.page_content)} characters\n")
                f.write(f"Metadata: {chunk.metadata}\n")
                f.write("-" * 30 + "\n")
                f.write(chunk.page_content)
                f.write("\n" + "=" * 50 + "\n\n")
        
        logger.info(f"Exported chunks to: {output_path}")
        return str(output_path)

def main():
    """
    Example usage of the DocumentProcessor
    """
    # Configuration for chunking
    config = ChunkingConfig(
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Initialize processor
    processor = DocumentProcessor(config)
    
    # Process a single file
    data_dir = Path(__file__).parent / "data"
    test_pdf = data_dir / "test.pdf"
    
    if test_pdf.exists():
        try:
            processed_doc = processor.process_file(str(test_pdf))
            print(f"Successfully processed: {processed_doc.filename}")
            print(f"File type: {processed_doc.file_type}")
            print(f"Total chunks: {processed_doc.total_chunks}")
            
            # Show preview of first few chunks
            previews = processor.get_chunk_preview(processed_doc.chunks, 2)
            print("\nChunk previews:")
            for preview in previews:
                print(f"Chunk {preview['chunk_index']}: {preview['content_length']} chars")
                print(f"Preview: {preview['content_preview'][:100]}...")
                print("-" * 40)
            
            # Export to text file for inspection
            export_path = processor.export_chunks_to_text(processed_doc)
            print(f"\nChunks exported to: {export_path}")
            
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print(f"Test file not found: {test_pdf}")
    
    # Process entire directory
    try:
        processed_docs = processor.process_directory(str(data_dir))
        print(f"\nProcessed {len(processed_docs)} documents from directory")
        for doc in processed_docs:
            print(f"- {doc.filename}: {doc.total_chunks} chunks")
    except Exception as e:
        print(f"Error processing directory: {e}")

if __name__ == "__main__":
    main()
