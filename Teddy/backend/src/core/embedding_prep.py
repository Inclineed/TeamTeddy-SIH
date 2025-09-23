"""
Example script showing how to prepare chunked documents for embedding generation
This demonstrates the next step after document processing - creating embeddings
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor, ChunkingConfig
from langchain.schema import Document


class EmbeddingPreparator:
    """
    Prepares chunked documents for embedding generation
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def prepare_chunks_for_embeddings(self, processed_doc) -> List[Dict[str, Any]]:
        """
        Prepare chunks in a format suitable for embedding generation
        
        Returns:
            List of dictionaries containing text and metadata for embeddings
        """
        embedding_data = []
        
        for i, chunk in enumerate(processed_doc.chunks):
            chunk_data = {
                "id": f"{processed_doc.filename}_{i}",
                "text": chunk.page_content,
                "metadata": {
                    "source_file": processed_doc.filename,
                    "chunk_index": i,
                    "total_chunks": processed_doc.total_chunks,
                    "file_type": processed_doc.file_type,
                    "chunk_size": len(chunk.page_content),
                    **chunk.metadata
                }
            }
            embedding_data.append(chunk_data)
        
        return embedding_data
    
    def export_for_embedding_service(self, embedding_data: List[Dict], output_path: str):
        """
        Export prepared data in JSON format for embedding services
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Embedding data exported to: {output_path}")
    
    def get_text_only(self, embedding_data: List[Dict]) -> List[str]:
        """
        Extract only the text content for embedding generation
        """
        return [item["text"] for item in embedding_data]
    
    def create_document_index(self, embedding_data: List[Dict]) -> Dict[str, Any]:
        """
        Create an index structure for the processed documents
        """
        files = {}
        
        for item in embedding_data:
            filename = item["metadata"]["source_file"]
            if filename not in files:
                files[filename] = {
                    "total_chunks": item["metadata"]["total_chunks"],
                    "file_type": item["metadata"]["file_type"],
                    "chunks": []
                }
            
            files[filename]["chunks"].append({
                "id": item["id"],
                "chunk_index": item["metadata"]["chunk_index"],
                "text_length": len(item["text"]),
                "text_preview": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
            })
        
        return {
            "total_files": len(files),
            "total_chunks": len(embedding_data),
            "files": files
        }


def demonstrate_embedding_preparation():
    """
    Demonstrate the complete pipeline from document to embedding-ready chunks
    """
    print("🔗 Document to Embedding Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize components
    config = ChunkingConfig(chunk_size=800, chunk_overlap=150)
    processor = DocumentProcessor(config)
    preparator = EmbeddingPreparator()
    
    # Find test documents
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        print("❌ No test files found. Please add PDF files to the data directory.")
        return
    
    all_embedding_data = []
    
    # Process each document
    for test_file in test_files:
        print(f"\n📄 Processing: {test_file.name}")
        
        try:
            # Step 1: Process document into chunks
            processed_doc = processor.process_file(str(test_file))
            print(f"✅ Created {processed_doc.total_chunks} chunks")
            
            # Step 2: Prepare chunks for embedding
            embedding_data = preparator.prepare_chunks_for_embeddings(processed_doc)
            all_embedding_data.extend(embedding_data)
            
            print(f"📊 Prepared {len(embedding_data)} items for embedding generation")
            
            # Show example of prepared data
            if embedding_data:
                example = embedding_data[0]
                print(f"📝 Example chunk:")
                print(f"  ID: {example['id']}")
                print(f"  Text length: {len(example['text'])} characters")
                print(f"  Text preview: {example['text'][:100]}...")
                print(f"  Metadata keys: {list(example['metadata'].keys())}")
            
        except Exception as e:
            print(f"❌ Error processing {test_file.name}: {e}")
            continue
    
    if all_embedding_data:
        # Export for embedding services
        output_path = data_dir / "embedding_ready_data.json"
        preparator.export_for_embedding_service(all_embedding_data, str(output_path))
        
        # Create document index
        doc_index = preparator.create_document_index(all_embedding_data)
        
        print(f"\n📋 Document Index Summary:")
        print(f"  Total files processed: {doc_index['total_files']}")
        print(f"  Total chunks created: {doc_index['total_chunks']}")
        
        for filename, file_info in doc_index['files'].items():
            print(f"  📄 {filename}: {file_info['total_chunks']} chunks ({file_info['file_type']})")
        
        # Export index
        index_path = data_dir / "document_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(doc_index, f, indent=2, ensure_ascii=False)
        print(f"📤 Document index exported to: {index_path}")
        
        # Get text-only list for embedding models
        texts_only = preparator.get_text_only(all_embedding_data)
        print(f"\n📝 Text-only list prepared with {len(texts_only)} items")
        
        print(f"\n🎯 Next Steps for Embedding Generation:")
        print("  1. Use the exported JSON data with your embedding model")
        print("  2. Generate embeddings for each text chunk")
        print("  3. Store embeddings with their corresponding metadata")
        print("  4. Build a vector database for similarity search")
        print("  5. Implement retrieval-augmented generation (RAG)")
        
        # Example of how to use with popular embedding libraries
        print(f"\n💡 Integration Examples:")
        print("  # With OpenAI embeddings:")
        print("  # embeddings = openai.Embedding.create(input=texts_only, model='text-embedding-ada-002')")
        print("  ")
        print("  # With Sentence Transformers:")
        print("  # from sentence_transformers import SentenceTransformer")
        print("  # model = SentenceTransformer('all-MiniLM-L6-v2')")
        print("  # embeddings = model.encode(texts_only)")
        print("  ")
        print("  # With Hugging Face transformers:")
        print("  # from transformers import AutoTokenizer, AutoModel")
        print("  # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')")
        print("  # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')")


def show_chunk_statistics():
    """
    Show detailed statistics about the chunking process
    """
    print("\n📊 Chunk Statistics Analysis")
    print("=" * 40)
    
    data_dir = Path(__file__).parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        return
    
    processor = DocumentProcessor()
    
    for test_file in test_files:
        try:
            processed_doc = processor.process_file(str(test_file))
            chunks = processed_doc.chunks
            
            if not chunks:
                continue
            
            # Calculate statistics
            chunk_lengths = [len(chunk.page_content) for chunk in chunks]
            word_counts = [len(chunk.page_content.split()) for chunk in chunks]
            
            print(f"\n📄 {processed_doc.filename}")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Character statistics:")
            print(f"    Min: {min(chunk_lengths)}")
            print(f"    Max: {max(chunk_lengths)}")
            print(f"    Average: {sum(chunk_lengths) / len(chunk_lengths):.0f}")
            print(f"  Word count statistics:")
            print(f"    Min: {min(word_counts)}")
            print(f"    Max: {max(word_counts)}")
            print(f"    Average: {sum(word_counts) / len(word_counts):.0f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {test_file.name}: {e}")


def main():
    """
    Main demonstration function
    """
    print("🚀 LangChain Document Processing → Embedding Pipeline")
    print("=" * 70)
    
    # Demonstrate the complete pipeline
    demonstrate_embedding_preparation()
    
    # Show chunk statistics
    show_chunk_statistics()
    
    print(f"\n✨ Pipeline demonstration completed!")
    print("Your documents are now chunked and ready for embedding generation! 🎉")


if __name__ == "__main__":
    main()