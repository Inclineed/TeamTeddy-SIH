"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    search_results: Optional[int] = Field(5)
    temperature: Optional[float] = Field(0.7)

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: bool
    response_time: float
    method: str

class ProcessedDocumentResponse(BaseModel):
    filename: str
    file_type: str
    total_chunks: int
    chunks_preview: List[Dict[str, Any]]
    processing_status: str
