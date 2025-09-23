"""
RAG Document Search API - Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

# Import organized API routers
from src.api.document_api import doc_app
from src.api.search_api import search_app
from src.api.rag_api import rag_app
from src.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="RAG Document Search API",
    description="A comprehensive RAG system for document search and question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include organized API routers
app.include_router(doc_app, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(search_app, prefix="/api/v1/search", tags=["Search"])
app.include_router(rag_app, prefix="/api/v1/rag", tags=["RAG"])

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "RAG Document Search API",
        "version": "1.0.0",
        "endpoints": {
            "documents": "/api/v1/documents",
            "search": "/api/v1/search", 
            "rag": "/api/v1/rag"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
