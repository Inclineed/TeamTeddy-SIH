"""
Pydantic models for request/response validation
Comprehensive schemas for all API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# Base schemas
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "success"

# Search related schemas
class SearchQuery(BaseModel):
    """Schema for search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    n_results: int = Field(5, ge=1, le=50, description="Number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

class SearchResult(BaseModel):
    """Schema for individual search results"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    source_file: str = Field(..., description="Source filename")
    chunk_index: int = Field(..., ge=0, description="Chunk index in document")

class SearchResponse(BaseResponse):
    """Schema for search responses"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results")
    search_time_ms: Optional[float] = Field(None, description="Search time in milliseconds")

# Document processing schemas
class ChunkingConfigSchema(BaseModel):
    """Schema for chunking configuration"""
    chunk_size: int = Field(1000, ge=100, le=4000, description="Size of each chunk")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Overlap between chunks")
    separator: str = Field("\n\n", description="Text separator for chunking")
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class DocumentChunk(BaseModel):
    """Schema for individual document chunks"""
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")

class ProcessedDocumentResponse(BaseResponse):
    """Schema for processed document responses"""
    filename: str = Field(..., description="Name of the processed file")
    file_type: str = Field(..., description="File type/extension")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks created")
    chunks_preview: List[Dict[str, Any]] = Field(..., description="Preview of first few chunks")
    metadata: Dict[str, Any] = Field(..., description="Document processing metadata")
    processing_status: str = Field(..., description="Processing status")

class DocumentListResponse(BaseResponse):
    """Schema for document list responses"""
    documents: List[ProcessedDocumentResponse] = Field(..., description="List of processed documents")
    total_documents: int = Field(..., ge=0, description="Total number of documents")

class UploadResponse(BaseResponse):
    """Schema for file upload responses"""
    original_filename: str = Field(..., description="Original filename")
    saved_filename: str = Field(..., description="Saved filename (may differ if duplicate)")
    file_path: str = Field(..., description="Path where file was saved")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    message: str = Field(..., description="Upload status message")

# RAG related schemas
class QuestionRequest(BaseModel):
    """Schema for question requests"""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    source_file: Optional[str] = Field(None, description="Optional specific file to search within")
    search_results: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature for answer generation")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

class MultipleQuestionsRequest(BaseModel):
    """Schema for multiple questions request"""
    questions: List[str] = Field(..., min_items=1, max_items=10, description="List of questions to answer")
    source_file: Optional[str] = Field(None, description="Optional specific file to search within")
    search_results: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature for answer generation")
    
    @validator('questions')
    def validate_questions(cls, v):
        cleaned_questions = []
        for q in v:
            if not q.strip():
                raise ValueError('Questions cannot be empty or whitespace only')
            cleaned_questions.append(q.strip())
        return cleaned_questions

class AnswerResponse(BaseResponse):
    """Schema for question answer responses"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source chunks used for context")
    context_used: bool = Field(..., description="Whether document context was used")
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    classification: Optional[Dict[str, Any]] = Field(None, description="Query classification info")
    method: str = Field(..., description="Method used to generate answer")
    max_relevance_score: Optional[float] = Field(None, description="Maximum relevance score of sources")
    num_sources: Optional[int] = Field(None, description="Number of sources used")

class MultipleAnswersResponse(BaseResponse):
    """Schema for multiple questions response"""
    questions_processed: int = Field(..., ge=0, description="Number of questions processed")
    results: List[AnswerResponse] = Field(..., description="Results for each question")
    total_response_time: float = Field(..., ge=0, description="Total response time in seconds")

# System status schemas
class ComponentStatus(BaseModel):
    """Schema for individual component status"""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional status details")

class SystemStatus(BaseResponse):
    """Schema for system status response"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    components: List[ComponentStatus] = Field(..., description="Status of individual components")
    overall_status: str = Field(..., description="Overall system status")

class CollectionStats(BaseModel):
    """Schema for collection statistics"""
    total_chunks: int = Field(..., ge=0, description="Total number of indexed chunks")
    collection_name: str = Field(..., description="Name of the collection")
    sample_source_files: List[str] = Field(..., description="Sample of source files")
    persist_directory: str = Field(..., description="Directory where collection is persisted")

# Configuration schemas
class RAGConfigSchema(BaseModel):
    """Schema for RAG configuration"""
    llm_model: str = Field(..., description="LLM model name")
    search_results: int = Field(..., ge=1, le=50, description="Number of search results")
    max_context_length: int = Field(..., ge=1000, le=16000, description="Maximum context length")
    temperature: float = Field(..., ge=0.0, le=2.0, description="LLM temperature")
    max_retries: int = Field(..., ge=1, le=10, description="Maximum number of retries")
    retry_delay: float = Field(..., ge=0.1, le=10.0, description="Delay between retries")

class EmbeddingConfigSchema(BaseModel):
    """Schema for embedding configuration"""
    model_name: str = Field(..., description="Embedding model name")
    batch_size: int = Field(..., ge=1, le=128, description="Batch size for embedding generation")
    max_retries: int = Field(..., ge=1, le=10, description="Maximum number of retries")
    retry_delay: float = Field(..., ge=0.1, le=10.0, description="Delay between retries")

# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses"""
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "error"
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ValidationError(BaseModel):
    """Schema for validation errors"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Invalid value")

class ValidationErrorResponse(ErrorResponse):
    """Schema for validation error responses"""
    error_type: str = "validation_error"
    errors: List[ValidationError] = Field(..., description="List of validation errors")

# Health check schemas
class HealthResponse(BaseModel):
    """Schema for health check responses"""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Service version")
    components: Optional[Dict[str, Any]] = Field(None, description="Component health details")

# Streaming response schemas (for documentation)
class StreamChunk(BaseModel):
    """Schema for streaming response chunks"""
    type: str = Field(..., description="Type of chunk (content, metadata, error, etc.)")
    content: Optional[str] = Field(None, description="Content chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata chunk")
    error: Optional[str] = Field(None, description="Error message")

# Index management schemas
class IndexRequest(BaseModel):
    """Schema for indexing requests"""
    filename: str = Field(..., description="Filename to index")
    chunk_size: int = Field(800, ge=100, le=4000, description="Size of each chunk")
    chunk_overlap: int = Field(150, ge=0, le=1000, description="Overlap between chunks")
    splitter_type: str = Field("recursive", description="Type of text splitter")
    
    @validator('splitter_type')
    def validate_splitter_type(cls, v):
        valid_types = ["recursive", "character", "token"]
        if v not in valid_types:
            raise ValueError(f'splitter_type must be one of: {valid_types}')
        return v

class IndexResponse(BaseResponse):
    """Schema for indexing responses"""
    filename: str = Field(..., description="Indexed filename")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks indexed")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
