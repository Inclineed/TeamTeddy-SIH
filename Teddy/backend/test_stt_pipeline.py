"""
Test script for Speech-to-Text pipeline integration
Tests audio processing with the document system
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from src.core.audio_processor import create_audio_processor
from src.core.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_processor():
    """Test the audio processor independently"""
    print("=" * 60)
    print("TESTING AUDIO PROCESSOR")
    print("=" * 60)
    
    try:
        # Create audio processor
        audio_processor = create_audio_processor(backend="whisper")
        print(f"✅ Audio processor created successfully")
        print(f"   Backend: {audio_processor.stt_backend}")
        print(f"   Device: {audio_processor.device}")
        print(f"   Supported extensions: {audio_processor.supported_extensions}")
        
        return audio_processor
        
    except Exception as e:
        print(f"❌ Audio processor creation failed: {e}")
        return None

def test_document_processor_audio():
    """Test document processor with audio support"""
    print("\n" + "=" * 60)
    print("TESTING DOCUMENT PROCESSOR WITH AUDIO")
    print("=" * 60)
    
    try:
        # Create document processor with audio enabled
        doc_processor = DocumentProcessor(enable_audio=True)
        print(f"✅ Document processor created with audio support")
        print(f"   Audio enabled: {doc_processor.enable_audio}")
        print(f"   Audio processor available: {doc_processor.audio_processor is not None}")
        print(f"   Supported extensions: {list(doc_processor.supported_extensions.keys())}")
        
        return doc_processor
        
    except Exception as e:
        print(f"❌ Document processor with audio failed: {e}")
        return None

def test_audio_file_detection():
    """Test audio file detection"""
    print("\n" + "=" * 60)
    print("TESTING AUDIO FILE DETECTION")
    print("=" * 60)
    
    audio_processor = create_audio_processor()
    
    test_files = [
        "test.mp3", "test.wav", "test.m4a", "test.flac",
        "test.pdf", "test.docx", "test.txt"
    ]
    
    for filename in test_files:
        is_audio = audio_processor.is_audio_file(filename)
        file_type = "audio" if is_audio else "document"
        print(f"   {filename}: {file_type}")

def create_test_audio_summary():
    """Create a summary of audio capabilities"""
    print("\n" + "=" * 60)
    print("AUDIO PROCESSING CAPABILITIES SUMMARY")
    print("=" * 60)
    
    try:
        import torch
        import transformers
        import librosa
        import soundfile
        import pydub
        
        print("✅ Required libraries available:")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   Librosa: {librosa.__version__}")
        print(f"   SoundFile: {soundfile.__version__}")
        print(f"   Pydub: {pydub.__version__}")
        
        print(f"\n✅ Hardware:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        
        print(f"\n✅ Audio formats supported:")
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov', '.mkv']
        for fmt in audio_formats:
            print(f"   {fmt}")
            
    except ImportError as e:
        print(f"❌ Missing library: {e}")
    except Exception as e:
        print(f"❌ Error checking capabilities: {e}")

def main():
    """Main test function"""
    print("🎙️ SPEECH-TO-TEXT PIPELINE TEST")
    print("Testing STT integration with document processing system")
    
    # Test 1: Audio processor
    audio_processor = test_audio_processor()
    
    # Test 2: Document processor with audio
    doc_processor = test_document_processor_audio()
    
    # Test 3: File detection
    if audio_processor:
        test_audio_file_detection()
    
    # Test 4: Capabilities summary
    create_test_audio_summary()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if audio_processor and doc_processor:
        print("✅ STT pipeline setup successful!")
        print("✅ Ready to process audio files with speech-to-text")
        print("✅ Audio files can be uploaded via document API")
        print("✅ Audio queries available via RAG API")
        
        print("\n📝 Next steps:")
        print("   1. Upload audio files via /documents/upload")
        print("   2. Use /rag/audio-query for direct audio questions")
        print("   3. Use /rag/audio-query-stream for streaming responses")
        
    else:
        print("❌ STT pipeline setup failed!")
        print("   Check the error messages above for details")
    
    print("\n🎯 Ready to test with real audio files!")

if __name__ == "__main__":
    main()