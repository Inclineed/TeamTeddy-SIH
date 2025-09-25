"""
Audio processing module for speech-to-text conversion
Supports various audio formats and STT models
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

import torch
import torchaudio
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np

# Import for Hugging Face Whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import for Azure Speech (alternative option)
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AudioTranscription:
    """Dataclass for audio transcription results"""
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class AudioProcessor:
    """
    Audio processor for speech-to-text conversion
    Supports multiple STT backends (Whisper, Azure Speech)
    """
    
    def __init__(self, 
                 stt_backend: str = "whisper",
                 whisper_model: str = "openai/whisper-small",
                 azure_speech_key: Optional[str] = None,
                 azure_speech_region: Optional[str] = None,
                 chunk_length_s: int = 30):
        """
        Initialize audio processor
        
        Args:
            stt_backend: STT backend to use ("whisper" or "azure")
            whisper_model: Whisper model name for Hugging Face
            azure_speech_key: Azure Speech API key
            azure_speech_region: Azure Speech region
            chunk_length_s: Max chunk length for processing (seconds)
        """
        self.stt_backend = stt_backend
        self.chunk_length_s = chunk_length_s
        
        # Supported audio formats
        self.supported_extensions = {
            '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', 
            '.wma', '.mp4', '.avi', '.mov', '.mkv'
        }
        
        # Initialize STT backend
        if stt_backend == "whisper":
            self._init_whisper(whisper_model)
        elif stt_backend == "azure" and AZURE_SPEECH_AVAILABLE:
            self._init_azure_speech(azure_speech_key, azure_speech_region)
        else:
            raise ValueError(f"Unsupported STT backend: {stt_backend}")
    
    def _init_whisper(self, model_name: str):
        """Initialize Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model.to(self.device)
            logger.info(f"Whisper model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _init_azure_speech(self, speech_key: str, speech_region: str):
        """Initialize Azure Speech Services"""
        if not AZURE_SPEECH_AVAILABLE:
            raise ImportError("Azure Cognitive Services Speech SDK not available")
        
        if not speech_key or not speech_region:
            raise ValueError("Azure Speech key and region required")
        
        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, 
            region=speech_region
        )
        self.speech_config.speech_recognition_language = "en-US"
        logger.info("Azure Speech Services initialized")
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is a supported audio format"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def _convert_to_wav(self, input_path: str, output_path: str) -> str:
        """Convert audio file to WAV format using pydub"""
        try:
            # Load audio with pydub (supports many formats)
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono, 16kHz (optimal for speech recognition)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            logger.info(f"Converted {input_path} to WAV format")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert audio to WAV: {e}")
            raise
    
    def _preprocess_audio(self, file_path: str) -> str:
        """Preprocess audio file for STT"""
        file_ext = Path(file_path).suffix.lower()
        
        # If already WAV, check if it needs resampling
        if file_ext == '.wav':
            try:
                # Check sample rate
                data, sample_rate = librosa.load(file_path, sr=None)
                if sample_rate == 16000 and len(data.shape) == 1:
                    return file_path  # Already optimal
            except Exception:
                pass
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        try:
            return self._convert_to_wav(file_path, temp_wav.name)
        except Exception:
            # Clean up temp file if conversion fails
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
            raise
    
    def _chunk_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Split audio into chunks for processing"""
        chunk_samples = self.chunk_length_s * sample_rate
        chunks = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) > sample_rate:  # Only process chunks > 1 second
                chunks.append(chunk)
        
        return chunks
    
    def _transcribe_with_whisper(self, audio_path: str) -> AudioTranscription:
        """Transcribe audio using Whisper"""
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(audio_data) / sample_rate
            
            # Process in chunks if audio is long
            if duration > self.chunk_length_s:
                chunks = self._chunk_audio(audio_data, sample_rate)
                transcriptions = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Prepare inputs
                    inputs = self.whisper_processor(
                        chunk, 
                        sampling_rate=sample_rate, 
                        return_tensors="pt"
                    )
                    inputs = inputs.to(self.device)
                    
                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.whisper_model.generate(
                            inputs["input_features"],
                            max_length=448,
                            num_beams=5,
                            early_stopping=True
                        )
                    
                    # Decode
                    transcription = self.whisper_processor.batch_decode(
                        predicted_ids, 
                        skip_special_tokens=True
                    )[0]
                    
                    if transcription.strip():
                        transcriptions.append(transcription.strip())
                
                final_text = " ".join(transcriptions)
            else:
                # Process entire audio at once
                inputs = self.whisper_processor(
                    audio_data, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    predicted_ids = self.whisper_model.generate(
                        inputs["input_features"],
                        max_length=448,
                        num_beams=5,
                        early_stopping=True
                    )
                
                final_text = self.whisper_processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
            
            return AudioTranscription(
                text=final_text.strip(),
                duration=duration,
                language="auto-detected",
                metadata={
                    "model": "whisper",
                    "sample_rate": sample_rate,
                    "chunks_processed": len(chunks) if duration > self.chunk_length_s else 1
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def _transcribe_with_azure(self, audio_path: str) -> AudioTranscription:
        """Transcribe audio using Azure Speech Services"""
        try:
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return AudioTranscription(
                    text=result.text,
                    confidence=result.confidence if hasattr(result, 'confidence') else None,
                    language=self.speech_config.speech_recognition_language,
                    metadata={"model": "azure_speech"}
                )
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                return AudioTranscription(text="", metadata={"model": "azure_speech", "reason": "no_speech"})
            else:
                raise Exception(f"Speech recognition failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure transcription failed: {e}")
            raise
    
    def transcribe_audio(self, file_path: str) -> AudioTranscription:
        """
        Main method to transcribe audio file to text
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioTranscription object with transcribed text and metadata
        """
        if not self.is_audio_file(file_path):
            raise ValueError(f"Unsupported audio format: {Path(file_path).suffix}")
        
        logger.info(f"Starting transcription of: {file_path}")
        
        # Preprocess audio
        processed_audio_path = self._preprocess_audio(file_path)
        temp_file_created = processed_audio_path != file_path
        
        try:
            # Transcribe based on backend
            if self.stt_backend == "whisper":
                result = self._transcribe_with_whisper(processed_audio_path)
            elif self.stt_backend == "azure":
                result = self._transcribe_with_azure(processed_audio_path)
            else:
                raise ValueError(f"Unknown STT backend: {self.stt_backend}")
            
            logger.info(f"Transcription completed. Text length: {len(result.text)} characters")
            return result
            
        finally:
            # Clean up temporary file if created
            if temp_file_created and os.path.exists(processed_audio_path):
                os.unlink(processed_audio_path)
    
    def process_audio_for_rag(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process audio file for RAG system integration
        
        Args:
            file_path: Path to audio file
            filename: Original filename
            
        Returns:
            Dictionary with transcribed text and metadata for RAG processing
        """
        transcription = self.transcribe_audio(file_path)
        
        # Prepare document-like structure for RAG integration
        return {
            "content": transcription.text,
            "metadata": {
                "filename": filename,
                "file_type": "audio",
                "original_extension": Path(file_path).suffix.lower(),
                "duration": transcription.duration,
                "language": transcription.language,
                "stt_backend": self.stt_backend,
                "confidence": transcription.confidence,
                **transcription.metadata
            }
        }

# Factory function for easy initialization
def create_audio_processor(backend: str = "whisper", **kwargs) -> AudioProcessor:
    """
    Factory function to create AudioProcessor instance
    
    Args:
        backend: STT backend ("whisper" or "azure")
        **kwargs: Additional arguments for AudioProcessor
        
    Returns:
        Configured AudioProcessor instance
    """
    return AudioProcessor(stt_backend=backend, **kwargs)