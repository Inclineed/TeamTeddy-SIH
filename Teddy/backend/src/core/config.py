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
    
    # LLM settings
    LLM_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    MAX_CONTEXT_LENGTH: int = 4000
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.UPLOAD_DIR, settings.EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
