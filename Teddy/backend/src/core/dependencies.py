# src/core/dependencies.py
from functools import lru_cache
from src.core.document_processor import DocumentProcessor
from src.core.semantic_search import SemanticSearchEngine

@lru_cache()
def get_document_processor():
    return DocumentProcessor()

@lru_cache() 
def get_search_engine():
    return SemanticSearchEngine()
