"""
Semantic Search System using ChromaDB and Ollama BGE-M3 Embeddings
Provides functionality to create embeddings for document chunks and perform semantic search
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import ollama
import numpy as np

from .document_processor import DocumentProcessor, ChunkingConfig, ProcessedDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "bge-m3"
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0

class OllamaEmbeddingGenerator:
    """
    Handles embedding generation using Ollama BGE-M3 model
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embedding generator
        
        Args:
            config: EmbeddingConfig object with embedding parameters
        """
        self.config = config or EmbeddingConfig()
        self.client = ollama.Client()
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the BGE-M3 model is available in Ollama"""
        try:
            models_response = self.client.list()
            
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
            
            # Check for exact match or with :latest suffix
            model_found = False
            for model_name in model_names:
                if (self.config.model_name == model_name or
                    f"{self.config.model_name}:latest" == model_name or
                    model_name.startswith(f"{self.config.model_name}:")):
                    model_found = True
                    break
            
            if not model_found:
                logger.warning(f"Model {self.config.model_name} not found in Ollama. Available models: {model_names}")
                logger.info(f"Please pull the model using: ollama pull {self.config.model_name}")
                
                # Try to pull the model automatically
                try:
                    logger.info(f"Attempting to pull {self.config.model_name} model...")
                    self.client.pull(self.config.model_name)
                    logger.info(f"Successfully pulled {self.config.model_name} model")
                except Exception as e:
                    logger.error(f"Failed to pull model {self.config.model_name}: {e}")
                    raise
            else:
                logger.info(f"Model {self.config.model_name} is available")
                
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            logger.info("Make sure Ollama is running and the model is available")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings(
                    model=self.config.model_name,
                    prompt=text
                )
                
                if 'embedding' in response:
                    return response['embedding']
                else:
                    raise ValueError(f"No embedding in response: {response}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for embedding generation: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts")
                    raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(texts) + self.config.batch_size - 1)//self.config.batch_size}")
            
            batch_embeddings = []
            for text in batch:
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid overwhelming Ollama
            time.sleep(0.1)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

class SemanticSearchEngine:
    """
    Main semantic search engine using ChromaDB and Ollama embeddings
    """
    
    def __init__(self,
                 collection_name: str = "document_chunks",
                 persist_directory: str = "./chroma_db",
                 embedding_config: EmbeddingConfig = None):
        """
        Initialize the semantic search engine
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_config: Configuration for embedding generation
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_generator = OllamaEmbeddingGenerator(embedding_config)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized SemanticSearchEngine with collection: {self.collection_name}")
        logger.info(f"Collection contains {self.collection.count()} documents")
    
    def add_documents(self, processed_doc: ProcessedDocument) -> int:
        """
        Add processed document chunks to the vector database
        
        Args:
            processed_doc: ProcessedDocument object with chunks
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Adding {processed_doc.total_chunks} chunks from {processed_doc.filename}")
        
        # Prepare data for ChromaDB
        texts = [chunk.page_content for chunk in processed_doc.chunks]
        ids = [f"{processed_doc.filename}_{i}" for i in range(len(texts))]
        
        # Prepare metadata
        metadatas = []
        for i, chunk in enumerate(processed_doc.chunks):
            metadata = {
                "source_file": processed_doc.filename,
                "chunk_index": i,
                "total_chunks": processed_doc.total_chunks,
                "file_type": processed_doc.file_type,
                "chunk_size": len(chunk.page_content),
                **{k: str(v) for k, v in chunk.metadata.items() if k not in ['source_file', 'chunk_index']}
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(texts)} chunks to the database")
            return len(texts)
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def search(self,
               query: str,
               n_results: int = 5,
               filter_metadata: Dict[str, Any] = None) -> List[SearchResult]:
        """
        Perform semantic search on the document collection
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: '{query}' (returning {n_results} results)")
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Perform search
        try:
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            results = []
            if search_results['documents'] and search_results['documents'][0]:
                documents = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                distances = search_results['distances'][0]
                ids = search_results['ids'][0]
                
                for i, (doc, metadata, distance, chunk_id) in enumerate(zip(documents, metadatas, distances, ids)):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1 - distance
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=doc,
                        score=score,
                        metadata=metadata,
                        source_file=metadata.get('source_file', 'unknown'),
                        chunk_index=int(metadata.get('chunk_index', 0))
                    )
                    
                    results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample to analyze
            sample = self.collection.peek(limit=10)
            
            # Analyze source files
            source_files = set()
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'source_file' in metadata:
                        source_files.add(metadata['source_file'])
            
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "sample_source_files": list(source_files),
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Get all IDs
            all_data = self.collection.get()
            
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents from collection")
            else:
                logger.info("Collection is already empty")
                
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

# Demo and testing functions
def demonstrate_semantic_search():
    """
    Demonstrate the complete semantic search pipeline
    """
    print("🔍 Semantic Search with ChromaDB and Ollama BGE-M3")
    print("=" * 60)
    
    # Initialize components
    config = ChunkingConfig(chunk_size=800, chunk_overlap=150)
    processor = DocumentProcessor(config)
    search_engine = SemanticSearchEngine(
        collection_name="demo_chunks",
        embedding_config=EmbeddingConfig(batch_size=16)
    )
    
    # Clear previous data for demo
    search_engine.clear_collection()
    
    # Find test documents
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_files = list(data_dir.glob("*.pdf"))
    
    if not test_files:
        print("❌ No test files found. Please add PDF files to the data directory.")
        return
    
    # Process and index documents
    print(f"\n📚 Indexing documents...")
    total_chunks = 0
    
    for test_file in test_files:
        print(f"\n📄 Processing: {test_file.name}")
        try:
            # Process document
            processed_doc = processor.process_file(str(test_file))
            print(f"✅ Created {processed_doc.total_chunks} chunks")
            
            # Add to search engine
            added_chunks = search_engine.add_documents(processed_doc)
            total_chunks += added_chunks
            print(f"🔗 Indexed {added_chunks} chunks")
            
        except Exception as e:
            print(f"❌ Error processing {test_file.name}: {e}")
            continue
    
    # Display collection stats
    stats = search_engine.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Source files: {stats.get('sample_source_files', [])}")
    
    # Demonstrate searches
    search_queries = [
        "What is meditation?",
        "How to achieve wisdom?",
        "What does Marcus Aurelius say about virtue?",
        "Stoic philosophy principles",
        "Death and mortality"
    ]
    
    print(f"\n🔍 Performing semantic searches...")
    for query in search_queries:
        print(f"\n🎯 Query: '{query}'")
        print("-" * 40)
        
        try:
            results = search_engine.search(query, n_results=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result.score:.3f}")
                    print(f"   Source: {result.source_file} (chunk {result.chunk_index})")
                    print(f"   Content: {result.content[:150]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    print(f"\n✨ Semantic search demonstration completed!")
    print(f"📈 You can now query {total_chunks} chunks from your documents!")

if __name__ == "__main__":
    demonstrate_semantic_search()
