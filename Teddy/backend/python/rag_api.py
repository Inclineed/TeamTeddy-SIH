"""
FastAPI endpoints for RAG Question-Answering system
Provides REST API for question answering with document context
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from rag_system import RAGQuestionAnswering, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
rag_app = FastAPI(title="RAG Question-Answering API", version="1.0.0")

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    source_file: Optional[str] = Field(None, description="Optional specific file to search within")
    search_results: Optional[int] = Field(5, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(0.7, description="LLM temperature for answer generation")

class MultipleQuestionsRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions to answer")
    source_file: Optional[str] = Field(None, description="Optional specific file to search within")

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: bool
    response_time: float
    classification: Optional[Dict[str, Any]] = None
    method: str
    max_relevance_score: Optional[float] = None
    num_sources: Optional[int] = None

# Global RAG system instance
rag_system = None

def get_rag_system() -> RAGQuestionAnswering:
    """Get or create RAG system instance"""
    global rag_system
    if rag_system is None:
        try:
            config = RAGConfig()
            rag_system = RAGQuestionAnswering(config)
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise HTTPException(status_code=500, detail=f"RAG system initialization failed: {str(e)}")
    return rag_system

@rag_app.get("/health")
async def health_check():
    """Health check endpoint for RAG system"""
    try:
        rag = get_rag_system()
        status = rag.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@rag_app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer with intelligent routing
    """
    try:
        rag = get_rag_system()
        
        # Update config if provided
        if request.temperature:
            rag.config.temperature = request.temperature
        
        logger.info(f"Processing question: '{request.question}'")
        
        result = await rag.answer_question(
            question=request.question,
            max_results=request.search_results or 5
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@rag_app.get("/ask")
async def ask_question_get(
    q: str = Query(..., description="The question to ask"),
    n_results: int = Query(5, description="Number of context chunks to retrieve"),
    temperature: float = Query(0.7, description="LLM temperature for answer generation")
):
    """
    Ask a question via GET request (convenient for testing)
    """
    try:
        request = QuestionRequest(
            question=q,
            search_results=n_results,
            temperature=temperature
        )
        return await ask_question(request)
        
    except Exception as e:
        logger.error(f"Error in GET question endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_app.post("/ask-multiple")
async def ask_multiple_questions(request: MultipleQuestionsRequest):
    """
    Ask multiple questions and get answers for all
    """
    try:
        rag = get_rag_system()
        
        logger.info(f"Processing {len(request.questions)} questions")
        
        results = await rag.answer_multiple_questions(request.questions)
        
        return JSONResponse(content={
            "questions_processed": len(request.questions),
            "results": results,
            "total_response_time": sum(r.get("response_time", 0) for r in results)
        })
        
    except Exception as e:
        logger.error(f"Error processing multiple questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process questions: {str(e)}")

@rag_app.get("/status")
async def get_system_status():
    """Get detailed status of RAG system components"""
    try:
        rag = get_rag_system()
        status = rag.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@rag_app.get("/config")
async def get_configuration():
    """Get current RAG system configuration"""
    try:
        rag = get_rag_system()
        config = {
            "llm_model": rag.config.llm_model,
            "search_results": rag.config.search_results,
            "max_context_length": rag.config.max_context_length,
            "temperature": rag.config.temperature,
            "max_retries": rag.config.max_retries,
            "retry_delay": rag.config.retry_delay
        }
        return JSONResponse(content=config)
    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_app.post("/config")
async def update_configuration(
    search_results: Optional[int] = Query(None, description="Number of search results"),
    temperature: Optional[float] = Query(None, description="LLM temperature"),
    max_context_length: Optional[int] = Query(None, description="Maximum context length")
):
    """Update RAG system configuration"""
    try:
        rag = get_rag_system()
        
        updates = {}
        if search_results is not None:
            rag.config.search_results = search_results
            updates["search_results"] = search_results
        if temperature is not None:
            rag.config.temperature = temperature
            updates["temperature"] = temperature
        if max_context_length is not None:
            rag.config.max_context_length = max_context_length
            updates["max_context_length"] = max_context_length
        
        return JSONResponse(content={
            "status": "updated",
            "changes": updates,
            "current_config": {
                "llm_model": rag.config.llm_model,
                "search_results": rag.config.search_results,
                "max_context_length": rag.config.max_context_length,
                "temperature": rag.config.temperature
            }
        })
        
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))