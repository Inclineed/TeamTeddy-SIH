"""
Test script to demonstrate document processing and chunking functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor, ChunkingConfig


def test_document_processing():
    """
    Test the document processing functionality
    """
    print("🚀 Testing Document Processing with LangChain")
    print("=" * 50)
    
    # Initialize processor with custom configuration
    config = ChunkingConfig(
        chunk_size=800,
        chunk_overlap=150
    )
    
    processor = DocumentProcessor(config)
    
    # Test with the existing PDF file
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        print("❌ No PDF files found in data directory")
        print("Please add some PDF files to test the functionality")
        return
    
    # Process each test file
    for test_file in test_files:
        print(f"\n📄 Processing: {test_file.name}")
        print("-" * 30)
        
        try:
            # Process the document
            processed_doc = processor.process_file(str(test_file))
            
            print(f"✅ Successfully processed: {processed_doc.filename}")
            print(f"📊 File type: {processed_doc.file_type}")
            print(f"📦 Total chunks: {processed_doc.total_chunks}")
            print(f"⚙️  Splitter used: {processed_doc.metadata['splitter_type']}")
            
            # Show chunk statistics
            if processed_doc.chunks:
                chunk_sizes = [len(chunk.page_content) for chunk in processed_doc.chunks]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                print(f"📏 Average chunk size: {avg_size:.0f} characters")
                print(f"📏 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")
            
            # Show preview of first few chunks
            print(f"\n🔍 Preview of chunks:")
            previews = processor.get_chunk_preview(processed_doc.chunks, 3)
            
            for i, preview in enumerate(previews):
                print(f"\nChunk {i + 1}:")
                print(f"  Length: {preview['content_length']} characters")
                print(f"  Preview: {preview['content_preview'][:100]}...")
                print(f"  Metadata: {preview['metadata']['source_file']}")
            
            # Export chunks for inspection
            export_path = processor.export_chunks_to_text(processed_doc)
            print(f"\n💾 Chunks exported to: {export_path}")
            
            # Test different chunking strategies
            print(f"\n🧪 Testing different chunking strategies:")
            
            for splitter_type in ["recursive", "character", "token"]:
                try:
                    chunks = processor.chunk_documents(
                        processor.load_document(str(test_file)), 
                        splitter_type
                    )
                    print(f"  {splitter_type}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"  {splitter_type}: ❌ Error - {e}")
            
        except Exception as e:
            print(f"❌ Error processing {test_file.name}: {e}")
            continue
    
    # Test directory processing
    print(f"\n📁 Testing directory processing...")
    try:
        processed_docs = processor.process_directory(str(data_dir))
        print(f"✅ Processed {len(processed_docs)} documents from directory")
        
        total_chunks = sum(doc.total_chunks for doc in processed_docs)
        print(f"📦 Total chunks across all documents: {total_chunks}")
        
        for doc in processed_docs:
            print(f"  - {doc.filename}: {doc.total_chunks} chunks ({doc.file_type})")
            
    except Exception as e:
        print(f"❌ Error processing directory: {e}")
    
    print(f"\n🎉 Document processing test completed!")
    print("=" * 50)


def test_chunking_configurations():
    """
    Test different chunking configurations
    """
    print("\n🔧 Testing Different Chunking Configurations")
    print("=" * 50)
    
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        print("❌ No test files available")
        return
    
    test_file = test_files[0]
    
    # Different configurations to test
    configs = [
        {"name": "Small chunks", "chunk_size": 500, "overlap": 50},
        {"name": "Medium chunks", "chunk_size": 1000, "overlap": 100},
        {"name": "Large chunks", "chunk_size": 2000, "overlap": 200},
        {"name": "No overlap", "chunk_size": 1000, "overlap": 0},
        {"name": "High overlap", "chunk_size": 1000, "overlap": 400}
    ]
    
    print(f"📄 Testing with file: {test_file.name}")
    
    for config_data in configs:
        config = ChunkingConfig(
            chunk_size=config_data["chunk_size"],
            chunk_overlap=config_data["overlap"]
        )
        
        processor = DocumentProcessor(config)
        
        try:
            processed_doc = processor.process_file(str(test_file))
            
            chunk_sizes = [len(chunk.page_content) for chunk in processed_doc.chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            print(f"\n📊 {config_data['name']}:")
            print(f"  Chunk size: {config_data['chunk_size']}, Overlap: {config_data['overlap']}")
            print(f"  Total chunks: {processed_doc.total_chunks}")
            print(f"  Average chunk size: {avg_size:.0f} characters")
            
        except Exception as e:
            print(f"❌ Error with {config_data['name']}: {e}")


def main():
    """
    Main test function
    """
    print("🧪 LangChain Document Processing Test Suite")
    print("=" * 60)
    
    # Test basic document processing
    test_document_processing()
    
    # Test different configurations
    test_chunking_configurations()
    
    print(f"\n✨ All tests completed!")
    print("\n💡 Next steps:")
    print("  1. The documents are now chunked and ready for embedding generation")
    print("  2. You can use the FastAPI endpoints to upload and process documents")
    print("  3. Start the server with: uvicorn server:app --reload")
    print("  4. Access the API docs at: http://localhost:8000/docs")
    print("  5. Document processing API at: http://localhost:8000/api/docs/docs")


if __name__ == "__main__":
    main()