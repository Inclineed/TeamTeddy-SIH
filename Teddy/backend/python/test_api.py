#!/usr/bin/env python3
"""
Test script for the integrated FastAPI server with document processing and semantic search
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_endpoints():
    """Test health endpoints for both APIs"""
    print("🔍 Testing Health Endpoints")
    print("=" * 50)
    
    # Test search API health
    try:
        response = requests.get(f"{BASE_URL}/api/search/health")
        print(f"✅ Search API Health: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Search API Health failed: {e}")
    
    # Test document API health
    try:
        response = requests.get(f"{BASE_URL}/api/documents/health")
        print(f"✅ Document API Health: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Document API Health failed: {e}")
    
    print()

def test_document_upload():
    """Test document upload and processing"""
    print("📄 Testing Document Upload")
    print("=" * 50)
    
    # Check if test file exists
    test_file = Path("data/test.pdf")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    try:
        # Upload document
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/documents/upload", files=files)
        
        print(f"✅ Document Upload: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   File: {result['filename']}")
            print(f"   Size: {result['size']} bytes")
            print(f"   Upload time: {result['upload_time']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Document upload failed: {e}")
    
    print()

def test_document_processing():
    """Test document processing"""
    print("⚙️ Testing Document Processing")
    print("=" * 50)
    
    try:
        # Process the uploaded document
        data = {
            "filename": "test.pdf",
            "chunking_strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        response = requests.post(f"{BASE_URL}/api/documents/process", json=data)
        print(f"✅ Document Processing: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Document: {result['document']['filename']}")
            print(f"   Total chunks: {result['document']['total_chunks']}")
            print(f"   Processing time: {result['processing_time']}ms")
            print(f"   Chunking strategy: {result['document']['chunking_config']['strategy']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
    
    print()

def test_embedding_generation():
    """Test embedding generation and indexing"""
    print("🔮 Testing Embedding Generation")
    print("=" * 50)
    
    try:
        # Generate embeddings and index
        data = {"filename": "test.pdf"}
        response = requests.post(f"{BASE_URL}/api/search/index", json=data)
        
        print(f"✅ Embedding Generation: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Indexed chunks: {result['indexed_chunks']}")
            print(f"   Collection: {result['collection_name']}")
            print(f"   Embedding time: {result['embedding_time']}ms")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
    
    print()

def test_semantic_search():
    """Test semantic search functionality"""
    print("🔍 Testing Semantic Search")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What is the meaning of life?",
        "How to achieve inner peace?",
        "What is wisdom according to philosophy?",
        "How to deal with adversity?"
    ]
    
    for query in test_queries:
        try:
            params = {"q": query, "n_results": 2}
            response = requests.get(f"{BASE_URL}/api/search/search", params=params)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query: '{query}'")
                print(f"   Results: {result['total_results']}")
                print(f"   Search time: {result['search_time_ms']:.2f}ms")
                
                # Show top result
                if result['results']:
                    top_result = result['results'][0]
                    print(f"   Top match (Score: {top_result['score']:.3f}): {top_result['content'][:100]}...")
                print()
            else:
                print(f"❌ Search failed for '{query}': {response.text}")
        except Exception as e:
            print(f"❌ Search failed for '{query}': {e}")

def test_collection_stats():
    """Test collection statistics"""
    print("📊 Testing Collection Statistics")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/api/search/stats")
        print(f"✅ Collection Stats: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Collection: {result['collection_name']}")
            print(f"   Storage: {result['persist_directory']}")
            if result['sample_source_files']:
                print(f"   Sample files: {result['sample_source_files'][:3]}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Stats request failed: {e}")
    
    print()

def main():
    """Run all API tests"""
    print("🚀 FastAPI Integration Test Suite")
    print("=" * 60)
    print()
    
    # Test all endpoints
    test_health_endpoints()
    test_document_upload()
    test_document_processing()
    test_embedding_generation()
    test_collection_stats()
    test_semantic_search()
    
    print("🎉 API Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()