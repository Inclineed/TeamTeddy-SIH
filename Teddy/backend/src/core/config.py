"""
Application configuration and settings
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "RAG Document Search API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "localhost"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    
    # Processing settings
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    COLLECTION_NAME: str = "document_chunks"
    
    # LLM settings (updated for Ollama)
    LLM_MODEL: str = "qwen2.5vl:7b"
    EMBEDDING_MODEL: str = "bge-m3"
    TEMPERATURE: float = 0.7
    MAX_CONTEXT_LENGTH: int = 4000
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 60
    
    # ChromaDB settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "document_chunks"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.UPLOAD_DIR, settings.EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create ChromaDB directory
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# Environment validation
def validate_environment():
    """Validate that required services are available"""
    import requests
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
        else:
            print("❌ Ollama service is not responding properly")
    except requests.RequestException:
        print("❌ Ollama service is not available. Please start Ollama.")
    
    # Check if required models are available
    try:
        import ollama
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        models = client.list()
        
        model_names = []
        if hasattr(models, 'models'):
            model_names = [model.model if hasattr(model, 'model') else model.get('name', '') for model in models.models]
        else:
            model_names = [model.get('name', model.get('model', '')) for model in models.get('models', [])]
        
        # Check LLM model
        llm_available = any(settings.LLM_MODEL in name for name in model_names)
        if llm_available:
            print(f"✅ LLM model {settings.LLM_MODEL} is available")
        else:
            print(f"❌ LLM model {settings.LLM_MODEL} not found. Available models: {model_names}")
            
        # Check embedding model
        embed_available = any(settings.EMBEDDING_MODEL in name for name in model_names)
        if embed_available:
            print(f"✅ Embedding model {settings.EMBEDDING_MODEL} is available")
        else:
            print(f"❌ Embedding model {settings.EMBEDDING_MODEL} not found. Available models: {model_names}")
            
    except Exception as e:
        print(f"❌ Failed to check Ollama models: {e}")

if __name__ == "__main__":
    validate_environment()
