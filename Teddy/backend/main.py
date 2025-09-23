"""
Main FastAPI application
Professional application assembly and configuration
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.api import documents, search, rag
from src.config import get_logger, get_settings, setup_logging
from src.core import DocumentProcessorError, EmbeddingServiceError, RAGSystemError, SearchServiceError

# Initialize configuration and logging
settings = get_settings()
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting RAG API Server...")
    
    # Startup logic
    try:
        # Validate critical configurations
        if not settings.ollama.base_url:
            raise ValueError("Ollama base URL not configured")
        
        # Initialize services in background to prevent blocking startup
        asyncio.create_task(_initialize_services())
        
        logger.info("RAG API Server started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    # Shutdown logic
    logger.info("Shutting down RAG API Server...")
    await _cleanup_services()
    logger.info("RAG API Server shutdown complete")


async def _initialize_services():
    """Initialize all services asynchronously"""
    try:
        # This will be called by dependency injection when needed
        # Services are initialized lazily to prevent startup blocking
        logger.info("Service initialization deferred to first request")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")


async def _cleanup_services():
    """Cleanup all services"""
    try:
        # Add cleanup logic if needed
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Service cleanup failed: {e}")


# Create FastAPI application
app = FastAPI(
    title="RAG API Server",
    description="Professional RAG (Retrieval-Augmented Generation) API with document processing and semantic search",
    version="1.0.0",
    docs_url="/docs" if settings.app.enable_docs else None,
    redoc_url="/redoc" if settings.app.enable_docs else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.app.allowed_hosts
)


# Global exception handlers
@app.exception_handler(DocumentProcessorError)
async def document_processor_exception_handler(request: Request, exc: DocumentProcessorError):
    """Handle document processor errors"""
    logger.error(f"Document processor error on {request.url}: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Document Processing Error",
            "detail": str(exc),
            "error_code": exc.error_code if hasattr(exc, 'error_code') else None
        }
    )


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_exception_handler(request: Request, exc: EmbeddingServiceError):
    """Handle embedding service errors"""
    logger.error(f"Embedding service error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Embedding Service Error",
            "detail": str(exc),
            "error_code": exc.error_code if hasattr(exc, 'error_code') else None
        }
    )


@app.exception_handler(SearchServiceError)
async def search_service_exception_handler(request: Request, exc: SearchServiceError):
    """Handle search service errors"""
    logger.error(f"Search service error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Search Service Error",
            "detail": str(exc),
            "error_code": exc.error_code if hasattr(exc, 'error_code') else None
        }
    )


@app.exception_handler(RAGSystemError)
async def rag_system_exception_handler(request: Request, exc: RAGSystemError):
    """Handle RAG system errors"""
    logger.error(f"RAG system error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "RAG System Error",
            "detail": str(exc),
            "error_code": exc.error_code if hasattr(exc, 'error_code') else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception on {request.url}: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred"
        }
    )


# Include API routers
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(rag.router, prefix="/api")


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs" if settings.app.enable_docs else None,
        "endpoints": {
            "documents": "/api/documents",
            "search": "/api/search",
            "rag": "/api/rag"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {}
        }
        
        # Add more detailed health checks here if needed
        # For now, just check if we're running
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/info")
async def app_info():
    """Application information endpoint"""
    return {
        "application": {
            "name": "RAG API Server",
            "version": "1.0.0",
            "environment": settings.app.environment,
            "debug": settings.app.debug
        },
        "configuration": {
            "host": settings.app.host,
            "port": settings.app.port,
            "cors_enabled": bool(settings.app.cors_origins),
            "docs_enabled": settings.app.enable_docs
        },
        "services": {
            "ollama": {
                "base_url": settings.ollama.base_url,
                "llm_model": settings.ollama.llm_model,
                "embedding_model": settings.ollama.embedding_model
            },
            "chroma": {
                "persist_directory": settings.chroma.persist_directory,
                "collection_name": settings.chroma.collection_name
            }
        }
    }


# Request middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = asyncio.get_event_loop().time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"(processed in {process_time:.3f}s)"
        )
        
        return response
        
    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url} "
            f"(failed after {process_time:.3f}s) - {e}"
        )
        raise


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_level="info" if not settings.app.debug else "debug",
        access_log=True
    )