from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import document API, search API, and RAG API (with error handling)
try:
    from document_api import doc_app
    DOC_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import document_api: {e}")
    DOC_API_AVAILABLE = False

try:
    from search_api import search_app
    SEARCH_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import search_api: {e}")
    SEARCH_API_AVAILABLE = False

try:
    from rag_api import rag_app
    RAG_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import rag_api: {e}")
    RAG_API_AVAILABLE = False

app = FastAPI(title="Team Teddy Backend API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the APIs if available
if DOC_API_AVAILABLE:
    app.mount("/api/docs", doc_app)

if SEARCH_API_AVAILABLE:
    app.mount("/api/search", search_app)

if RAG_API_AVAILABLE:
    app.mount("/api/rag", rag_app)

@app.get("/")
def root():
    services = {
        "greet": "/greet/{name}"
    }
    if DOC_API_AVAILABLE:
        services.update({
            "document_processing": "/api/docs",
            "document_health": "/api/docs/health"
        })
    
    if SEARCH_API_AVAILABLE:
        services.update({
            "semantic_search": "/api/search",
            "search_health": "/api/search/health"
        })
    
    if RAG_API_AVAILABLE:
        services.update({
            "rag_question_answering": "/api/rag",
            "rag_health": "/api/rag/health",
            "ask_question": "/api/rag/ask"
        })
    
    return {
        "message": "Team Teddy Backend API",
        "services": services,
        "doc_api_status": "available" if DOC_API_AVAILABLE else "unavailable",
        "search_api_status": "available" if SEARCH_API_AVAILABLE else "unavailable",
        "rag_api_status": "available" if RAG_API_AVAILABLE else "unavailable"
    }@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello {name}, from FastAPI!"}
