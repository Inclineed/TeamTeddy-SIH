"""
FastAPI endpoints for RAG Question-Answering system
Provides REST API for question answering with document context
"""

from fastapi import APIRouter, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import logging
import json
import tempfile
import os
from pathlib import Path

from src.core.rag_system import RAGQuestionAnswering, RAGConfig
from src.core.audio_processor import create_audio_processor
from src.core.multimodal_processor import create_multimodal_processor
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIRouter
rag_app = APIRouter()

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

class AudioQueryResponse(BaseModel):
    transcribed_text: str
    answer: str
    sources: List[Dict[str, Any]]
    context_used: bool
    response_time: float
    audio_metadata: Dict[str, Any]
    classification: Optional[Dict[str, Any]] = None
    method: str

class MultimodalQueryResponse(BaseModel):
    image_analysis: str
    answer: str
    sources: List[Dict[str, Any]]
    context_used: bool
    response_time: float
    image_metadata: Dict[str, Any]
    text_query: str
    method: str

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

@rag_app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question and get a streaming answer with intelligent routing
    Returns server-sent events with chunks of the answer as it's generated
    """
    try:
        rag = get_rag_system()
        
        # Update config if provided
        if request.temperature:
            rag.config.temperature = request.temperature
            
        logger.info(f"Processing streaming question: '{request.question}'")
        
        async def generate_stream() -> AsyncGenerator[str, None]:
            """Generate SSE stream for the answer"""
            try:
                async for chunk in rag.answer_question_stream(
                    question=request.question,
                    max_results=request.search_results or 5
                ):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send end of stream marker
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                # Send error in SSE format
                error_chunk = {
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to setup streaming: {str(e)}")

@rag_app.get("/ask-stream")
async def ask_question_stream_get(
    q: str = Query(..., description="The question to ask"),
    n_results: int = Query(5, description="Number of context chunks to retrieve"),
    temperature: float = Query(0.7, description="LLM temperature for answer generation")
):
    """
    Ask a question via GET request and get streaming answer (convenient for testing)
    """
    try:
        request = QuestionRequest(
            question=q,
            search_results=n_results,
            temperature=temperature
        )
        
        return await ask_question_stream(request)
        
    except Exception as e:
        logger.error(f"Error in GET streaming question endpoint: {e}")
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

@rag_app.post("/audio-query", response_model=AudioQueryResponse)
async def audio_query(
    audio_file: UploadFile = File(..., description="Audio file to transcribe and query"),
    additional_text: Optional[str] = Form(None, description="Additional text to combine with transcribed audio"),
    source_file: Optional[str] = Form(None, description="Optional specific file to search within"),
    search_results: Optional[int] = Form(5, description="Number of context chunks to retrieve"),
    temperature: Optional[float] = Form(0.7, description="LLM temperature for answer generation")
):
    """
    Speech-to-Text + RAG endpoint: Transcribe audio and answer questions
    
    This endpoint:
    1. Accepts an audio file upload
    2. Transcribes the audio to text using Whisper STT
    3. Optionally combines with additional text query
    4. Uses the transcribed text as a question for RAG system
    5. Returns both transcription and answer with sources
    """
    import time
    start_time = time.time()
    
    try:
        # Validate audio file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Check audio file extension
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov', '.mkv']
        file_extension = Path(audio_file.filename).suffix.lower()
        
        if file_extension not in audio_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_extension}. Supported: {', '.join(audio_extensions)}"
            )
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            # Save uploaded audio to temp file
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize audio processor
            logger.info(f"Processing audio file: {audio_file.filename}")
            audio_processor = create_audio_processor(backend="whisper")
            
            # Transcribe audio
            transcription_result = audio_processor.transcribe_audio(temp_file_path)
            transcribed_text = transcription_result.text
            
            if not transcribed_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="No speech detected in audio file"
                )
            
            logger.info(f"Audio transcribed successfully: {len(transcribed_text)} characters")
            
            # Combine transcribed text with additional text if provided
            final_question = transcribed_text
            if additional_text and additional_text.strip():
                final_question = f"{transcribed_text} {additional_text}"
            
            # Get RAG system and answer the question
            rag = get_rag_system()
            
            # Temporarily update temperature if provided
            original_temperature = rag.config.temperature
            if temperature is not None:
                rag.config.temperature = temperature
            
            try:
                # Get answer using RAG system
                answer_result = await rag.answer_question(
                    final_question,
                    max_results=search_results
                )
            finally:
                # Restore original temperature
                rag.config.temperature = original_temperature
            
            # Calculate total response time
            response_time = time.time() - start_time
            
            # Prepare audio metadata
            audio_metadata = {
                "filename": audio_file.filename,
                "file_extension": file_extension,
                "transcription_duration": transcription_result.duration,
                "transcription_language": transcription_result.language,
                "transcription_confidence": transcription_result.confidence,
                "stt_backend": "whisper",
                "character_count": len(transcribed_text),
                "combined_with_text": bool(additional_text and additional_text.strip())
            }
            
            # Add transcription metadata if available
            if transcription_result.metadata:
                audio_metadata.update(transcription_result.metadata)
            
            return AudioQueryResponse(
                transcribed_text=transcribed_text,
                answer=answer_result["answer"],
                sources=answer_result["sources"],
                context_used=answer_result["context_used"],
                response_time=response_time,
                audio_metadata=audio_metadata,
                classification=answer_result.get("classification"),
                method=answer_result.get("method", "rag")
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio query processing failed: {str(e)}")

@rag_app.post("/audio-query-stream")
async def audio_query_stream(
    audio_file: UploadFile = File(..., description="Audio file to transcribe and query"),
    additional_text: Optional[str] = Form(None, description="Additional text to combine with transcribed audio"),
    source_file: Optional[str] = Form(None, description="Optional specific file to search within"),
    search_results: Optional[int] = Form(5, description="Number of context chunks to retrieve"),
    temperature: Optional[float] = Form(0.7, description="LLM temperature for answer generation")
):
    """
    Streaming Speech-to-Text + RAG endpoint
    
    This endpoint:
    1. Accepts an audio file upload
    2. Transcribes the audio to text using Whisper STT
    3. Optionally combines with additional text query
    4. Uses the transcribed text as a question for RAG system
    5. Streams the response in real-time
    """
    try:
        # Validate audio file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Check audio file extension
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov', '.mkv']
        file_extension = Path(audio_file.filename).suffix.lower()
        
        if file_extension not in audio_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_extension}. Supported: {', '.join(audio_extensions)}"
            )
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            # Save uploaded audio to temp file
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        async def generate_audio_stream():
            try:
                # Send initial status
                yield "data: " + json.dumps({
                    "type": "status",
                    "message": "Processing audio file...",
                    "audio_filename": audio_file.filename
                }) + "\n\n"
                
                # Initialize audio processor
                logger.info(f"Processing audio file: {audio_file.filename}")
                audio_processor = create_audio_processor(backend="whisper")
                
                # Transcribe audio
                yield "data: " + json.dumps({
                    "type": "status",
                    "message": "Transcribing audio..."
                }) + "\n\n"
                
                transcription_result = audio_processor.transcribe_audio(temp_file_path)
                transcribed_text = transcription_result.text
                
                if not transcribed_text.strip():
                    yield "data: " + json.dumps({
                        "type": "error",
                        "error": "No speech detected in audio file"
                    }) + "\n\n"
                    return
                
                # Send transcription result
                yield "data: " + json.dumps({
                    "type": "transcription",
                    "transcribed_text": transcribed_text,
                    "audio_metadata": {
                        "filename": audio_file.filename,
                        "file_extension": file_extension,
                        "transcription_duration": transcription_result.duration,
                        "transcription_language": transcription_result.language,
                        "transcription_confidence": transcription_result.confidence,
                        "character_count": len(transcribed_text)
                    }
                }) + "\n\n"
                
                logger.info(f"Audio transcribed successfully: {len(transcribed_text)} characters")
                
                # Combine transcribed text with additional text if provided
                final_question = transcribed_text
                if additional_text and additional_text.strip():
                    final_question = f"{transcribed_text} {additional_text}"
                    
                    yield "data: " + json.dumps({
                        "type": "status",
                        "message": "Combined transcription with additional text",
                        "final_question": final_question
                    }) + "\n\n"
                
                # Get RAG system and stream the answer
                yield "data: " + json.dumps({
                    "type": "status",
                    "message": "Generating answer..."
                }) + "\n\n"
                
                rag = get_rag_system()
                
                # Temporarily update temperature if provided
                original_temperature = rag.config.temperature
                if temperature is not None:
                    rag.config.temperature = temperature
                
                try:
                    # Stream the answer using RAG system
                    async for chunk in rag.answer_question_stream(
                        final_question,
                        max_results=search_results
                    ):
                        # Forward the RAG streaming chunks
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                finally:
                    # Restore original temperature
                    rag.config.temperature = original_temperature
                
                # Send completion
                yield "data: " + json.dumps({
                    "type": "complete",
                    "method": "audio_rag_stream"
                }) + "\n\n"
                
            except Exception as e:
                logger.error(f"Audio streaming query failed: {e}")
                yield "data: " + json.dumps({
                    "type": "error",
                    "error": f"Audio query processing failed: {str(e)}"
                }) + "\n\n"
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio streaming setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio streaming setup failed: {str(e)}")

@rag_app.post("/multimodal-query", response_model=MultimodalQueryResponse)
async def multimodal_query(
    image_file: UploadFile = File(..., description="Image file to analyze"),
    text_query: str = Form(..., description="Text question about the image"),
    search_results: Optional[int] = Form(5, description="Number of context chunks to retrieve"),
    temperature: Optional[float] = Form(0.7, description="LLM temperature for answer generation"),
    use_document_context: Optional[bool] = Form(True, description="Whether to include document context in analysis")
):
    """
    Multimodal RAG endpoint: Analyze image with text query
    
    This endpoint:
    1. Accepts an image file upload and text query
    2. Analyzes the image using Qwen2.5VL vision model
    3. Optionally retrieves relevant document context
    4. Combines image analysis with text query and context
    5. Returns comprehensive multimodal response
    """
    import time
    start_time = time.time()
    
    try:
        # Validate image file
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Check image file extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        file_extension = Path(image_file.filename).suffix.lower()
        
        if file_extension not in image_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {file_extension}. Supported: {', '.join(image_extensions)}"
            )
        
        # Create temporary file for image processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            # Save uploaded image to temp file
            content = await image_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize multimodal processor
            logger.info(f"Processing multimodal query - Image: {image_file.filename}, Query: {text_query[:100]}...")
            multimodal_processor = create_multimodal_processor(model="llava:7b")
            
            # Get document context if requested
            context_documents = []
            if use_document_context:
                try:
                    rag = get_rag_system()
                    # Search for relevant documents using the text query
                    search_results_data = rag.search_engine.search(text_query, n_results=search_results)
                    context_documents = search_results_data if search_results_data else []
                    logger.info(f"Retrieved {len(context_documents)} context documents")
                except Exception as e:
                    logger.warning(f"Failed to retrieve document context: {e}")
                    context_documents = []
            
            # Analyze image with context
            multimodal_result = multimodal_processor.analyze_image_with_context(
                temp_file_path,
                text_query,
                context_documents
            )
            
            # Calculate total response time
            response_time = time.time() - start_time
            
            # Prepare sources from context documents
            sources = []
            if context_documents:
                sources = [
                    {
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "metadata": doc.metadata,
                        "relevance_score": doc.score
                    }
                    for doc in context_documents[:search_results]
                ]
            
            logger.info(f"Multimodal query completed: {len(multimodal_result['answer'])} characters")
            
            return MultimodalQueryResponse(
                image_analysis=multimodal_result["image_analysis"],
                answer=multimodal_result["answer"],
                sources=sources,
                context_used=multimodal_result["context_used"],
                response_time=response_time,
                image_metadata=multimodal_result["image_metadata"],
                text_query=text_query,
                method=multimodal_result["method"]
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal query processing failed: {str(e)}")

@rag_app.post("/multimodal-query-stream")
async def multimodal_query_stream(
    image_file: UploadFile = File(..., description="Image file to analyze"),
    text_query: str = Form(..., description="Text question about the image"),
    search_results: Optional[int] = Form(5, description="Number of context chunks to retrieve"),
    temperature: Optional[float] = Form(0.7, description="LLM temperature for answer generation"),
    use_document_context: Optional[bool] = Form(True, description="Whether to include document context in analysis")
):
    """
    Streaming Multimodal RAG endpoint
    
    Similar to multimodal-query but streams the response for real-time interaction
    """
    try:
        # Validate inputs
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        if not text_query:
            raise HTTPException(status_code=400, detail="Text query is required for multimodal analysis")
        
        # Check image file extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        file_extension = Path(image_file.filename).suffix.lower()
        
        if file_extension not in image_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {file_extension}. Supported: {', '.join(image_extensions)}"
            )
        
        # Create temporary file for image processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            content = await image_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        async def generate_multimodal_response():
            try:
                # Initialize multimodal processor
                logger.info(f"Processing streaming multimodal query - Image: {image_file.filename}")
                multimodal_processor = create_multimodal_processor(model="llava:7b")
                
                # First, send image analysis
                yield "data: " + json.dumps({
                    "type": "status",
                    "message": "Analyzing image...",
                    "image_filename": image_file.filename
                }) + "\n\n"
                
                # Get basic image analysis
                image_result = multimodal_processor.analyze_image(
                    temp_file_path, 
                    prompt=f"Analyze this image in the context of this question: {text_query}"
                )
                
                yield "data: " + json.dumps({
                    "type": "image_analysis",
                    "image_analysis": image_result.description,
                    "image_metadata": image_result.metadata
                }) + "\n\n"
                
                # Get document context if requested
                context_documents = []
                if use_document_context:
                    yield "data: " + json.dumps({
                        "type": "status",
                        "message": "Retrieving relevant documents..."
                    }) + "\n\n"
                    
                    try:
                        rag = get_rag_system()
                        search_results_data = rag.search_engine.search(text_query, n_results=search_results)
                        context_documents = search_results_data if search_results_data else []
                        
                        yield "data: " + json.dumps({
                            "type": "context",
                            "num_documents": len(context_documents),
                            "context_used": len(context_documents) > 0
                        }) + "\n\n"
                        
                    except Exception as e:
                        logger.warning(f"Failed to retrieve document context: {e}")
                        yield "data: " + json.dumps({
                            "type": "warning",
                            "message": f"Could not retrieve document context: {str(e)}"
                        }) + "\n\n"
                
                # Generate final response
                yield "data: " + json.dumps({
                    "type": "status",
                    "message": "Generating comprehensive response..."
                }) + "\n\n"
                
                # Build comprehensive prompt for streaming
                prompt_parts = [
                    f"User Question: {text_query}",
                    f"Image Analysis: {image_result.description}"
                ]
                
                if context_documents:
                    context_text = "\n\n".join([
                        f"Document {i+1}: {doc.content[:500]}..."
                        for i, doc in enumerate(context_documents[:3])
                    ])
                    prompt_parts.append(f"Relevant Context:\n{context_text}")
                
                prompt_parts.append(
                    "Based on the image analysis and any provided context, "
                    "provide a comprehensive answer to the user's question."
                )
                
                full_prompt = "\n\n".join(prompt_parts)
                
                # Stream the final response
                response_stream = multimodal_processor.ollama_client.chat(
                    model="llava:7b",
                    messages=[
                        {
                            'role': 'user',
                            'content': full_prompt,
                            'images': [multimodal_processor._image_to_base64(
                                multimodal_processor._preprocess_image(temp_file_path)[0]
                            )]
                        }
                    ],
                    stream=True
                )
                
                # Stream response chunks
                for chunk in response_stream:
                    if chunk['message'].get('content'):
                        yield "data: " + json.dumps({
                            "type": "answer_chunk",
                            "content": chunk['message']['content']
                        }) + "\n\n"
                
                # Send completion
                yield "data: " + json.dumps({
                    "type": "complete",
                    "method": "multimodal_rag_stream"
                }) + "\n\n"
                
            except Exception as e:
                logger.error(f"Streaming multimodal query failed: {e}")
                yield "data: " + json.dumps({
                    "type": "error",
                    "error": f"Multimodal query processing failed: {str(e)}"
                }) + "\n\n"
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        return StreamingResponse(
            generate_multimodal_response(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal streaming setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal streaming setup failed: {str(e)}")
