from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import document API (with error handling)
try:
    from document_api import doc_app
    DOC_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import document_api: {e}")
    DOC_API_AVAILABLE = False

app = FastAPI(title="Team Teddy Backend API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the document processing API if available
if DOC_API_AVAILABLE:
    app.mount("/api/docs", doc_app)

@app.get("/")
def root():
    services = {
        "greet": "/greet/{name}"
    }
    if DOC_API_AVAILABLE:
        services.update({
            "document_processing": "/api/docs",
            "health": "/api/docs/health"
        })
    
    return {
        "message": "Team Teddy Backend API", 
        "services": services,
        "doc_api_status": "available" if DOC_API_AVAILABLE else "unavailable"
    }

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello {name}, from FastAPI!"}
