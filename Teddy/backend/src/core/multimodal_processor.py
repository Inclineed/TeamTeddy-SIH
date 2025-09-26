"""
Multimodal processing module for handling images with text
Supports image analysis and multimodal RAG queries
"""

import os
import base64
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import ollama

logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    description: str
    metadata: Dict[str, Any]
    base64_image: Optional[str] = None
    processed_image_path: Optional[str] = None

class MultimodalProcessor:
    """
    Processor for handling images and multimodal content
    Integrates with Qwen2.5VL model for image understanding
    """
    
    def __init__(self, 
                 vision_model: str = "llava:7b",
                 max_image_size: Tuple[int, int] = (1024, 1024),
                 supported_formats: Optional[List[str]] = None):
        """
        Initialize multimodal processor
        
        Args:
            vision_model: Ollama vision model name
            max_image_size: Maximum image dimensions (width, height)
            supported_formats: Supported image formats
        """
        self.vision_model = vision_model
        self.max_image_size = max_image_size
        self.ollama_client = ollama.Client()
        
        # Supported image formats
        self.supported_formats = supported_formats or [
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'
        ]
        
        # Verify model availability
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the vision model is available"""
        try:
            models = self.ollama_client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.vision_model not in available_models:
                logger.warning(f"Vision model {self.vision_model} not found. Available models: {available_models}")
            else:
                logger.info(f"Vision model {self.vision_model} is available")
                
        except Exception as e:
            logger.error(f"Failed to verify vision model: {e}")
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to fit within max dimensions while maintaining aspect ratio"""
        if image.size[0] <= self.max_image_size[0] and image.size[1] <= self.max_image_size[1]:
            return image
        
        # Calculate new size maintaining aspect ratio
        ratio = min(self.max_image_size[0] / image.size[0], 
                   self.max_image_size[1] / image.size[1])
        
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def _preprocess_image(self, image_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """Preprocess image for analysis"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get original metadata
            metadata = {
                "original_size": image.size,
                "format": image.format or "Unknown",
                "mode": image.mode,
                "filename": Path(image_path).name
            }
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                metadata["converted_to_rgb"] = True
            
            # Resize if needed
            original_size = image.size
            image = self._resize_image(image)
            
            if image.size != original_size:
                metadata["resized"] = True
                metadata["new_size"] = image.size
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_image(self, 
                     image_path: str, 
                     prompt: str = "Describe this image in detail",
                     include_base64: bool = False) -> ImageAnalysisResult:
        """
        Analyze image using vision model
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            include_base64: Whether to include base64 encoded image
            
        Returns:
            ImageAnalysisResult with description and metadata
        """
        try:
            if not self.is_image_file(image_path):
                raise ValueError(f"Unsupported image format: {Path(image_path).suffix}")
            
            # Preprocess image
            image, metadata = self._preprocess_image(image_path)
            
            # Convert to base64 for model
            base64_image = self._image_to_base64(image)
            
            # Analyze with vision model
            logger.info(f"Analyzing image with {self.vision_model}: {Path(image_path).name}")
            
            response = self.ollama_client.chat(
                model=self.vision_model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_image]
                    }
                ]
            )
            
            description = response['message']['content']
            
            # Update metadata
            metadata.update({
                "analysis_prompt": prompt,
                "model_used": self.vision_model,
                "description_length": len(description)
            })
            
            logger.info(f"Image analysis completed: {len(description)} characters")
            
            return ImageAnalysisResult(
                description=description,
                metadata=metadata,
                base64_image=base64_image if include_base64 else None
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed for {image_path}: {e}")
            raise
    
    def analyze_image_with_context(self, 
                                  image_path: str, 
                                  text_query: str,
                                  context_documents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze image with text query and optional document context
        
        Args:
            image_path: Path to image file
            text_query: Text question/query about the image
            context_documents: Optional relevant documents for context
            
        Returns:
            Dictionary with image analysis and contextual response
        """
        try:
            # First, get image description
            image_result = self.analyze_image(
                image_path, 
                prompt=f"Analyze this image in the context of this question: {text_query}"
            )
            
            # Build comprehensive prompt
            prompt_parts = [
                f"User Question: {text_query}",
                f"Image Analysis: {image_result.description}"
            ]
            
            # Add document context if available
            if context_documents:
                context_text = "\n\n".join([
                    f"Document {i+1}: {doc.get('content', '')[:500]}..."
                    for i, doc in enumerate(context_documents[:3])
                ])
                prompt_parts.append(f"Relevant Context:\n{context_text}")
            
            # Add instruction
            prompt_parts.append(
                "Based on the image analysis and any provided context, "
                "provide a comprehensive answer to the user's question."
            )
            
            full_prompt = "\n\n".join(prompt_parts)
            
            # Get final response from model
            response = self.ollama_client.chat(
                model=self.vision_model,
                messages=[
                    {
                        'role': 'user',
                        'content': full_prompt,
                        'images': [self._image_to_base64(
                            self._preprocess_image(image_path)[0]
                        )]
                    }
                ]
            )
            
            final_answer = response['message']['content']
            
            return {
                "answer": final_answer,
                "image_analysis": image_result.description,
                "image_metadata": image_result.metadata,
                "context_used": bool(context_documents),
                "num_context_docs": len(context_documents) if context_documents else 0,
                "method": "multimodal_rag"
            }
            
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            raise
    
    def process_image_for_rag(self, 
                             image_path: str, 
                             filename: str,
                             description_prompt: str = "Provide a detailed description of this image") -> Dict[str, Any]:
        """
        Process image for RAG system integration
        
        Args:
            image_path: Path to image file
            filename: Original filename
            description_prompt: Prompt for image description
            
        Returns:
            Dictionary with image description and metadata for RAG processing
        """
        try:
            # Analyze image
            analysis_result = self.analyze_image(image_path, description_prompt)
            
            # Prepare document-like structure for RAG integration
            return {
                "content": analysis_result.description,
                "metadata": {
                    "filename": filename,
                    "file_type": "image",
                    "original_extension": Path(image_path).suffix.lower(),
                    "vision_model": self.vision_model,
                    "analysis_prompt": description_prompt,
                    **analysis_result.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Image processing for RAG failed: {e}")
            raise

# Factory function for easy initialization
def create_multimodal_processor(model: str = "llava:7b", **kwargs) -> MultimodalProcessor:
    """
    Factory function to create MultimodalProcessor instance
    
    Args:
        model: Vision model name
        **kwargs: Additional arguments for MultimodalProcessor
        
    Returns:
        Configured MultimodalProcessor instance
    """
    return MultimodalProcessor(vision_model=model, **kwargs)