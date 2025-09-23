"""
Complete Semantic Search Demo
Demonstrates the full pipeline from document processing to semantic search
"""

import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor, ChunkingConfig
from semantic_search import SemanticSearchEngine, EmbeddingConfig

async def complete_demo():
    """
    Complete demonstration of the semantic search system
    """
    print("🚀 Complete Semantic Search Demo")
    print("=" * 50)
    
    # Step 1: Check Ollama and model availability
    print("\n📋 Step 1: Checking Prerequisites")
    print("-" * 30)
    
    try:
        from semantic_search import OllamaEmbeddingGenerator
        embedding_gen = OllamaEmbeddingGenerator()
        print("✅ Ollama connection successful")
        print("✅ BGE-M3 model available")
        
        # Test embedding generation
        test_embedding = embedding_gen.generate_embedding("test")
        print(f"✅ Embedding generation working ({len(test_embedding)} dimensions)")
        
    except Exception as e:
        print(f"❌ Prerequisites check failed: {e}")
        print("\n💡 Please ensure:")
        print("  1. Ollama is running")
        print("  2. BGE-M3 model is installed: ollama pull bge-m3")
        return
    
    # Step 2: Document Processing
    print("\n📄 Step 2: Processing Documents")
    print("-" * 30)
    
    config = ChunkingConfig(chunk_size=600, chunk_overlap=100)
    processor = DocumentProcessor(config)
    
    # Find test documents
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        print("❌ No PDF files found in data directory")
        print("Please add some PDF files to test the functionality")
        return
    
    processed_docs = []
    for test_file in test_files:
        try:
            print(f"📖 Processing: {test_file.name}")
            processed_doc = processor.process_file(str(test_file))
            processed_docs.append(processed_doc)
            print(f"   ✅ Created {processed_doc.total_chunks} chunks")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if not processed_docs:
        print("❌ No documents were successfully processed")
        return
    
    # Step 3: Initialize Search Engine
    print("\n🔍 Step 3: Initializing Search Engine")
    print("-" * 30)
    
    try:
        search_engine = SemanticSearchEngine(
            collection_name="demo_semantic_search",
            embedding_config=EmbeddingConfig(batch_size=8)
        )
        
        # Clear previous data
        search_engine.clear_collection()
        print("✅ Search engine initialized")
        print("✅ Collection cleared for fresh demo")
        
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return
    
    # Step 4: Index Documents
    print("\n📚 Step 4: Indexing Documents")
    print("-" * 30)
    
    total_indexed = 0
    for processed_doc in processed_docs:
        try:
            print(f"🔗 Indexing: {processed_doc.filename}")
            indexed_count = search_engine.add_documents(processed_doc)
            total_indexed += indexed_count
            print(f"   ✅ Indexed {indexed_count} chunks")
        except Exception as e:
            print(f"   ❌ Indexing failed: {e}")
            continue
    
    print(f"\n📊 Total indexed chunks: {total_indexed}")
    
    # Step 5: Collection Statistics
    print("\n📈 Step 5: Collection Statistics")
    print("-" * 30)
    
    stats = search_engine.get_collection_stats()
    print(f"📦 Total chunks: {stats['total_chunks']}")
    print(f"🗃️  Collection: {stats['collection_name']}")
    print(f"📁 Storage: {stats['persist_directory']}")
    
    # Step 6: Semantic Search Demonstration
    print("\n🎯 Step 6: Semantic Search Demonstration")
    print("-" * 30)
    
    # Test queries
    test_queries = [
        "What is the meaning of life?",
        "How should one deal with adversity?",
        "What does it mean to be virtuous?",
        "How to achieve inner peace?",
        "What is the nature of death?",
        "How to control emotions?",
        "What is wisdom according to philosophy?",
        "How to live a good life?"
    ]
    
    print(f"🔍 Testing {len(test_queries)} semantic queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎯 Query {i}: '{query}'")
        print("─" * 50)
        
        try:
            results = search_engine.search(query, n_results=3)
            
            if results:
                for j, result in enumerate(results, 1):
                    print(f"\n   {j}. Score: {result.score:.3f}")
                    print(f"      Source: {result.source_file}")
                    print(f"      Chunk: {result.chunk_index}/{result.metadata.get('total_chunks', 'N/A')}")
                    
                    # Show content preview
                    content_preview = result.content[:200].replace('\n', ' ')
                    if len(result.content) > 200:
                        content_preview += "..."
                    print(f"      Content: {content_preview}")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
        
        # Small delay between queries
        await asyncio.sleep(0.5)
    
    # Step 7: Advanced Search Features
    print("\n🔬 Step 7: Advanced Search Features")
    print("-" * 30)
    
    # Search with metadata filters
    if processed_docs:
        sample_file = processed_docs[0].filename
        print(f"🔍 Searching within specific file: {sample_file}")
        
        try:
            filtered_results = search_engine.search(
                "wisdom and virtue", 
                n_results=2,
                filter_metadata={"source_file": sample_file}
            )
            
            print(f"   Found {len(filtered_results)} results in {sample_file}")
            for result in filtered_results:
                print(f"   • Score: {result.score:.3f} | Chunk {result.chunk_index}")
                
        except Exception as e:
            print(f"   ❌ Filtered search failed: {e}")
    
    # Step 8: Performance Analysis
    print("\n⚡ Step 8: Performance Analysis")
    print("-" * 30)
    
    import time
    
    # Measure search performance
    performance_query = "philosophy and wisdom"
    search_times = []
    
    for i in range(5):
        start_time = time.time()
        results = search_engine.search(performance_query, n_results=5)
        search_time = (time.time() - start_time) * 1000
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    
    print(f"🏎️  Average search time: {avg_search_time:.2f}ms")
    print(f"📊 Search times: {[f'{t:.1f}ms' for t in search_times]}")
    print(f"🎯 Results per search: {len(results)}")
    
    # Step 9: Summary and Next Steps
    print("\n🎉 Step 9: Demo Summary")
    print("-" * 30)
    
    print(f"✅ Successfully processed {len(processed_docs)} documents")
    print(f"✅ Indexed {total_indexed} text chunks")
    print(f"✅ Performed semantic searches with {avg_search_time:.1f}ms average response time")
    print(f"✅ ChromaDB collection persisted at: {stats['persist_directory']}")
    
    print(f"\n🚀 Next Steps:")
    print("  1. Start the FastAPI server: uvicorn server:app --reload")
    print("  2. Access the API docs at: http://localhost:8000/docs")
    print("  3. Use semantic search API at: http://localhost:8000/api/search")
    print("  4. Try search queries via: GET /api/search/search?q=your_query")
    print("  5. Check collection stats: GET /api/search/stats")
    
    print(f"\n📝 API Examples:")
    print("  • Search: curl 'http://localhost:8000/api/search/search?q=wisdom&n_results=3'")
    print("  • Stats: curl 'http://localhost:8000/api/search/stats'")
    print("  • Health: curl 'http://localhost:8000/api/search/health'")
    
    print(f"\n✨ Semantic search system is ready for use!")

def quick_search_test():
    """
    Quick test function for semantic search
    """
    print("🔍 Quick Semantic Search Test")
    print("=" * 40)
    
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            collection_name="document_chunks",
            embedding_config=EmbeddingConfig(batch_size=4)
        )
        
        # Get stats
        stats = search_engine.get_collection_stats()
        print(f"📊 Collection has {stats['total_chunks']} chunks")
        
        if stats['total_chunks'] == 0:
            print("❌ No documents indexed. Please run the complete demo first.")
            return
        
        # Test search
        query = "What is wisdom and virtue?"
        print(f"\n🎯 Searching for: '{query}'")
        
        results = search_engine.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f}")
            print(f"   Source: {result.source_file}")
            print(f"   Content: {result.content[:150]}...")
        
        print(f"\n✅ Search test completed!")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")

def main():
    """
    Main function to run the demo
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search Demo")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick search test instead of full demo"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        quick_search_test()
    else:
        asyncio.run(complete_demo())

if __name__ == "__main__":
    main()