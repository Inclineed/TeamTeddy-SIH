"""
Dependency injection for FastAPI endpoints
Provides singleton instances of core services
"""

from functools import lru_cache
import logging

from src.core.document_processor import DocumentProcessor, ChunkingConfig
from src.core.semantic_search import SemanticSearchEngine, EmbeddingConfig
from src.core.rag_system import RAGQuestionAnswering, RAGConfig
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """
    Get singleton DocumentProcessor instance
    """
    try:
        config = ChunkingConfig(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        processor = DocumentProcessor(config)
        logger.info("DocumentProcessor initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize DocumentProcessor: {e}")
        raise

@lru_cache()
def get_search_engine() -> SemanticSearchEngine:
    """
    Get singleton SemanticSearchEngine instance
    """
    try:
        embedding_config = EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=16
        )
        
        search_engine = SemanticSearchEngine(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_config=embedding_config
        )
        logger.info("SemanticSearchEngine initialized successfully")
        return search_engine
    except Exception as e:
        logger.error(f"Failed to initialize SemanticSearchEngine: {e}")
        raise

@lru_cache()
def get_rag_system() -> RAGQuestionAnswering:
    """
    Get singleton RAG system instance
    """
    try:
        config = RAGConfig(
            llm_model=settings.LLM_MODEL,
            search_results=5,
            max_context_length=settings.MAX_CONTEXT_LENGTH,
            temperature=settings.TEMPERATURE
        )
        
        rag_system = RAGQuestionAnswering(config)
        logger.info("RAG system initialized successfully")
        return rag_system
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

# Health check functions
def check_dependencies() -> dict:
    """
    Check the health of all dependencies
    """
    status = {
        "document_processor": "unknown",
        "search_engine": "unknown", 
        "rag_system": "unknown",
        "overall": "unknown"
    }
    
    try:
        # Test document processor
        processor = get_document_processor()
        status["document_processor"] = "healthy"
    except Exception as e:
        status["document_processor"] = f"error: {str(e)}"
    
    try:
        # Test search engine
        search_engine = get_search_engine()
        search_stats = search_engine.get_collection_stats()
        status["search_engine"] = "healthy"
        status["search_engine_stats"] = search_stats
    except Exception as e:
        status["search_engine"] = f"error: {str(e)}"
    
    try:
        # Test RAG system
        rag = get_rag_system()
        rag_status = rag.get_system_status()
        status["rag_system"] = "healthy"
        status["rag_system_status"] = rag_status
    except Exception as e:
        status["rag_system"] = f"error: {str(e)}"
    
    # Overall status
    errors = [v for v in status.values() if isinstance(v, str) and v.startswith("error:")]
    if not errors:
        status["overall"] = "healthy"
    else:
        status["overall"] = f"errors: {len(errors)}"
    
    return status

# Cleanup functions for graceful shutdown
def cleanup_dependencies():
    """
    Cleanup all dependencies on shutdown
    """
    try:
        # Clear LRU caches
        get_document_processor.cache_clear()
        get_search_engine.cache_clear()
        get_rag_system.cache_clear()
        logger.info("Dependencies cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
