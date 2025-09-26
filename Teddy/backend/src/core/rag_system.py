"""
RAG Question-Answering System using Ollama phi3:mini
Combines semantic search with LLM to provide contextual answers
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import ollama

from .semantic_search import SemanticSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    llm_model: str = "phi3:mini"
    search_results: int = 5
    max_context_length: int = 4000
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_query_classification: bool = True
    context_threshold: float = 0.5  # Minimum relevance score to use context
    direct_answer_categories: List[str] = None
    
    def __post_init__(self):
        if self.direct_answer_categories is None:
            self.direct_answer_categories = [
                "general_knowledge", "mathematics", "science", "weather",
                "current_events", "simple_facts", "calculations"
            ]

class RAGQuestionAnswering:
    """
    RAG (Retrieval-Augmented Generation) system that combines semantic search
    with Ollama phi3:mini for generating contextual answers
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize RAG system with semantic search and LLM"""
        self.config = config or RAGConfig()
        self.ollama_client = ollama.Client()
        
        # Initialize semantic search engine
        self.search_engine = SemanticSearchEngine()
        
        # Check if phi3:mini model is available
        self._check_llm_availability()
        
        logger.info(f"RAG system initialized with model: {self.config.llm_model}")
        logger.info(f"Query classification enabled: {self.config.enable_query_classification}")
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify whether a query needs document context or can be answered directly
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary with classification results
        """
        if not self.config.enable_query_classification:
            return {"needs_context": True, "confidence": 1.0, "reason": "classification_disabled"}
        
        # Create classification prompt
        classification_prompt = f"""You are a query classifier. Determine if this question needs specific document context or can be answered with general knowledge.

Question: "{query}"

Classify this question into one of these categories:
1. NEEDS_CONTEXT: Question about specific documents, texts, or domain-specific information
2. GENERAL_KNOWLEDGE: Question that can be answered with general knowledge (math, science, weather, current events, simple facts)

Consider these examples:
- "What is the weather today?" → GENERAL_KNOWLEDGE
- "What is 2+2?" → GENERAL_KNOWLEDGE
- "What does this document say about virtue?" → NEEDS_CONTEXT
- "According to the text, how should one live?" → NEEDS_CONTEXT
- "What is machine learning?" → GENERAL_KNOWLEDGE
- "How does the author define wisdom?" → NEEDS_CONTEXT

Respond with only: NEEDS_CONTEXT or GENERAL_KNOWLEDGE"""
        
        try:
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": classification_prompt}],
                options={"temperature": 0.1}  # Low temperature for consistent classification
            )
            
            classification = response['message']['content'].strip().upper()
            
            if "GENERAL_KNOWLEDGE" in classification:
                return {
                    "needs_context": False,
                    "confidence": 0.8,
                    "reason": "classified_as_general_knowledge",
                    "category": "general_knowledge"
                }
            elif "NEEDS_CONTEXT" in classification:
                return {
                    "needs_context": True,
                    "confidence": 0.8,
                    "reason": "classified_as_context_dependent",
                    "category": "context_dependent"
                }
            else:
                # If unclear, default to using context
                return {
                    "needs_context": True,
                    "confidence": 0.3,
                    "reason": "unclear_classification_default_to_context",
                    "category": "unclear"
                }
                
        except Exception as e:
            logger.warning(f"Query classification failed: {e}. Defaulting to context search.")
            return {
                "needs_context": True,
                "confidence": 0.0,
                "reason": f"classification_error: {str(e)}",
                "category": "error"
            }
    
    def _generate_direct_answer(self, query: str) -> str:
        """Generate a direct answer without document context"""
        direct_prompt = f"""{query}"""
        
        try:
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": direct_prompt}],
                options={
                    "temperature": self.config.temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Direct answer generation failed: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    def _check_llm_availability(self):
        """Check if phi3:mini model is available in Ollama"""
        try:
            models_response = self.ollama_client.list()
            
            # Handle both dictionary and object responses
            if hasattr(models_response, 'models'):
                models = models_response.models
            else:
                models = models_response.get('models', [])
            
            # Extract model names
            model_names = []
            for model in models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                else:
                    model_names.append(model.get('name', model.get('model', '')))
            
            # Check for exact match or with suffix
            model_found = False
            for model_name in model_names:
                if (self.config.llm_model == model_name or
                    f"{self.config.llm_model}:latest" == model_name or
                    model_name.startswith(f"{self.config.llm_model}:")):
                    model_found = True
                    break
            
            if not model_found:
                logger.warning(f"Model {self.config.llm_model} not found. Available models: {model_names}")
                logger.info(f"Please pull the model using: ollama pull {self.config.llm_model}")
                
                # Try to pull the model automatically
                try:
                    logger.info(f"Attempting to pull {self.config.llm_model} model...")
                    self.ollama_client.pull(self.config.llm_model)
                    logger.info(f"Successfully pulled {self.config.llm_model} model")
                except Exception as e:
                    logger.error(f"Failed to pull model {self.config.llm_model}: {e}")
                    raise
            else:
                logger.info(f"Model {self.config.llm_model} is available")
                
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            logger.info("Make sure Ollama is running and the model is available")
            raise
    
    def _create_prompt(self, query: str, context_chunks: List) -> str:
        """Create a prompt for the LLM with context from retrieved documents"""
        # Format context from search results
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.source_file
            filename = chunk.metadata.get('filename', 'Unknown')
            content = chunk.content
            score = chunk.score
            
            context_text += f"[Context {i}] (Relevance: {score:.3f})\n"
            context_text += f"Source: {filename}\n"
            context_text += f"Content: {content}\n\n"
        
        # Truncate context if too long
        if len(context_text) > self.config.max_context_length:
            context_text = context_text[:self.config.max_context_length] + "...\n[Context truncated]"
        
        # Create the prompt
        prompt = f"""You are an intelligent assistant that answers questions based on provided context from documents.

CONTEXT FROM DOCUMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based PRIMARILY on the provided context from the documents
2. If the context contains relevant information, use it to provide a comprehensive answer
3. If the context doesn't fully address the question, mention this limitation
4. Be specific and reference the document content when possible
5. If multiple perspectives are present in the context, acknowledge them
6. Keep your answer clear, informative, and well-structured

ANSWER:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using Ollama phi3:mini model"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.ollama_client.chat(
                    model=self.config.llm_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    options={
                        'temperature': self.config.temperature,
                        'top_p': 0.9,
                        'top_k': 40
                    }
                )
                
                if 'message' in response and 'content' in response['message']:
                    return response['message']['content'].strip()
                else:
                    raise ValueError(f"Unexpected response format: {response}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for answer generation: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to generate answer after {self.config.max_retries} attempts")
                    raise
    
    def _generate_answer_stream(self, prompt: str):
        """Generate streaming answer using Ollama phi3:mini model"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.ollama_client.chat(
                    model=self.config.llm_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    options={
                        'temperature': self.config.temperature,
                        'top_p': 0.9,
                        'top_k': 40
                    },
                    stream=True  # Enable streaming
                )
                
                # Yield each chunk of the response
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        if content:  # Only yield non-empty content
                            yield content
                
                return  # Exit successfully after streaming
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for streaming answer generation: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to generate streaming answer after {self.config.max_retries} attempts")
                    raise
    
    async def answer_question(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG with intelligent query classification
        
        Args:
            question: The question to answer
            max_results: Maximum number of chunks to retrieve for context
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Classify the query
            classification = self._classify_query_intent(question)
            logger.info(f"Query classification: {classification}")
            
            # Step 2: Route based on classification
            if not classification["needs_context"]:
                # Generate direct answer without document context
                logger.info("Generating direct answer without document context")
                answer = self._generate_direct_answer(question)
                
                return {
                    "answer": answer,
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "classification": classification,
                    "method": "direct_answer"
                }
            
            # Step 3: Proceed with RAG pipeline for context-dependent queries
            logger.info("Using RAG pipeline with document context")
            
            # Perform semantic search
            search_results = self.search_engine.search(
                query=question,
                n_results=max_results,
                filter_metadata=None
            )
            
            if not search_results:
                # No relevant context found, provide direct answer with note
                logger.warning("No relevant context found, falling back to direct answer")
                answer = self._generate_direct_answer(question)
                answer += "\n\n*Note: This answer is based on general knowledge as no relevant information was found in the provided documents.*"
                
                return {
                    "answer": answer,
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "classification": classification,
                    "method": "fallback_direct_answer",
                    "fallback_reason": "no_relevant_context"
                }
            
            # Step 4: Evaluate context relevance
            max_score = max(result.score for result in search_results)
            
            if max_score < self.config.context_threshold:
                # Context not relevant enough, provide direct answer with note
                logger.info(f"Context relevance too low (max_score: {max_score:.3f} < threshold: {self.config.context_threshold})")
                answer = self._generate_direct_answer(question)
                answer += f"\n\n*Note: This answer is based on general knowledge as the document context was not sufficiently relevant (relevance score: {max_score:.2f}).*"
                
                return {
                    "answer": answer,
                    "sources": [{"content": result.content, "score": result.score, "metadata": result.metadata}
                               for result in search_results[:3]],  # Include some context for reference
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "classification": classification,
                    "method": "low_relevance_direct_answer",
                    "max_relevance_score": max_score
                }
            
            # Step 5: Generate answer with document context
            logger.info(f"Using document context (max relevance: {max_score:.3f})")
            
            # Create prompt with context
            prompt = self._create_prompt(question, search_results)
            
            # Generate answer
            answer = self._generate_answer(prompt)
            
            # Prepare sources
            sources = []
            for result in search_results:
                sources.append({
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata
                })
            
            response_time = time.time() - start_time
            logger.info(f"Generated answer in {response_time:.2f} seconds")
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "response_time": response_time,
                "classification": classification,
                "method": "rag_with_context",
                "max_relevance_score": max_score,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            # Fallback to direct answer on error
            try:
                fallback_answer = self._generate_direct_answer(question)
                return {
                    "answer": f"I encountered an error while processing your question, but here's what I can tell you based on general knowledge:\n\n{fallback_answer}",
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "classification": {"needs_context": True, "confidence": 0.0, "reason": "error_fallback"},
                    "method": "error_fallback",
                    "error": str(e)
                }
            except Exception as fallback_error:
                return {
                    "answer": f"I apologize, but I encountered errors while processing your question: {str(e)}",
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "classification": {"needs_context": True, "confidence": 0.0, "reason": "error"},
                    "method": "error",
                    "error": str(e),
                    "fallback_error": str(fallback_error)
                }
    
    async def answer_question_stream(self, question: str, max_results: int = 5):
        """
        Stream answer generation for a question using RAG with intelligent query classification
        """
        start_time = time.time()
        
        try:
            # Step 1: Classify the query
            classification = self._classify_query_intent(question)
            logger.info(f"Query classification: {classification}")
            
            # Yield initial metadata
            yield {
                "type": "metadata",
                "classification": classification,
                "method": "rag_with_context" if classification["needs_context"] else "direct_answer"
            }
            
            # Step 2: Route based on classification
            if not classification["needs_context"]:
                # Generate direct answer without document context
                logger.info("Generating direct answer without document context")
                prompt = f"""You are a helpful AI assistant. Answer the following question directly and concisely:

Question: {question}

Answer:"""
                
                yield {"type": "method", "method": "direct_answer"}
                
                for chunk in self._generate_answer_stream(prompt):
                    yield {"type": "content", "content": chunk}
                
                yield {
                    "type": "final_metadata",
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "method": "direct_answer"
                }
                return
            
            # Step 3: Proceed with RAG pipeline for context-dependent queries
            logger.info("Using RAG pipeline with document context")
            
            # Perform semantic search
            search_results = self.search_engine.search(
                query=question,
                n_results=max_results,
                filter_metadata=None
            )
            
            if not search_results:
                # No relevant context found, provide direct answer with note
                logger.warning("No relevant context found, falling back to direct answer")
                prompt = f"""You are a helpful AI assistant. Answer the following question directly, and mention that this is based on general knowledge as no relevant documents were found:

Question: {question}

Answer:"""
                
                yield {"type": "method", "method": "fallback_direct_answer"}
                
                for chunk in self._generate_answer_stream(prompt):
                    yield {"type": "content", "content": chunk}
                
                yield {
                    "type": "final_metadata",
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "method": "fallback_direct_answer",
                    "fallback_reason": "no_relevant_context"
                }
                return
            
            # Step 4: Generate answer with document context
            max_score = max(result.score for result in search_results)
            logger.info(f"Using document context (max relevance: {max_score:.3f})")
            
            # Prepare context and sources
            prompt = self._create_prompt(question, search_results)
            sources = [{"content": result.content, "score": result.score, "metadata": result.metadata}
                      for result in search_results]
            
            # Yield sources information
            yield {"type": "sources", "sources": sources[:3]}  # Limit to first 3 sources
            yield {"type": "method", "method": "rag_with_context"}
            
            # Generate streaming response
            for chunk in self._generate_answer_stream(prompt):
                yield {"type": "content", "content": chunk}
            
            yield {
                "type": "final_metadata",
                "sources": sources,
                "context_used": True,
                "response_time": time.time() - start_time,
                "method": "rag_with_context",
                "max_relevance_score": max_score,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question_stream: {e}")
            
            # Fallback to direct answer on error
            try:
                prompt = f"""You are a helpful AI assistant. I encountered an error while processing the question, but answer based on general knowledge:

Question: {question}

Answer:"""
                
                yield {"type": "method", "method": "error_fallback"}
                yield {"type": "content", "content": "I encountered an error while processing your question, but here's what I can tell you based on general knowledge:\n\n"}
                
                for chunk in self._generate_answer_stream(prompt):
                    yield {"type": "content", "content": chunk}
                
                yield {
                    "type": "final_metadata",
                    "sources": [],
                    "context_used": False,
                    "response_time": time.time() - start_time,
                    "method": "error_fallback",
                    "error": str(e)
                }
                
            except Exception as fallback_error:
                yield {
                    "type": "error",
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "response_time": time.time() - start_time
                }
    
    async def answer_multiple_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple questions in batch"""
        results = []
        for question in questions:
            logger.info(f"Processing question: {question}")
            result = await self.answer_question(question)
            results.append(result)
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of RAG system components"""
        try:
            # Check search engine status
            search_stats = self.search_engine.get_collection_stats()
            
            # Check LLM availability
            llm_available = True
            try:
                test_response = self.ollama_client.chat(
                    model=self.config.llm_model,
                    messages=[{'role': 'user', 'content': 'Hello'}]
                )
                llm_status = "healthy"
            except Exception as e:
                llm_available = False
                llm_status = f"error: {str(e)}"
            
            return {
                "rag_system": "healthy",
                "search_engine": {
                    "status": "healthy",
                    "indexed_chunks": search_stats.get("total_chunks", 0),
                    "collection": search_stats.get("collection_name", "unknown")
                },
                "llm": {
                    "model": self.config.llm_model,
                    "status": llm_status,
                    "available": llm_available
                },
                "configuration": {
                    "search_results": self.config.search_results,
                    "max_context_length": self.config.max_context_length,
                    "temperature": self.config.temperature
                }
            }
            
        except Exception as e:
            return {
                "rag_system": "error",
                "error": str(e)
            }

# Factory function for easy initialization
def create_rag_system(config: RAGConfig = None) -> RAGQuestionAnswering:
    """Create and return a RAG system instance"""
    return RAGQuestionAnswering(config)
