import requests
import json
import time

def test_streaming_with_general_knowledge():
    """Test streaming API with a general knowledge question that doesn't require documents"""
    
    url = "http://localhost:8000/api/v1/rag/ask-stream"
    
    # This question should trigger direct answer mode since it's general knowledge
    payload = {
        "question": "What is the meaning of life and existence?",
        "search_results": 5,
        "temperature": 0.7
    }
    
    print("🤖 Testing Streaming RAG API")
    print("="*50)
    print(f"Question: {payload['question']}")
    print("\n🔄 Streaming response:")
    print("-"*30)
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        full_answer = ""
        metadata = {}
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                
                if data_str == "[DONE]":
                    print("\n\n✅ Stream completed!")
                    break
                
                try:
                    chunk = json.loads(data_str)
                    
                    if chunk["type"] == "metadata":
                        needs_context = chunk.get("classification", {}).get("needs_context", "unknown")
                        method = chunk.get("method", "unknown")
                        print(f"🧠 Classification: needs_context={needs_context}, method={method}")
                        
                    elif chunk["type"] == "method":
                        print(f"🛠️  Method: {chunk['method']}")
                        
                    elif chunk["type"] == "content":
                        content = chunk["content"]
                        print(content, end="", flush=True)
                        full_answer += content
                        
                    elif chunk["type"] == "final_metadata":
                        metadata = chunk
                        
                    elif chunk["type"] == "error":
                        print(f"\n❌ Error: {chunk['error']}")
                        return
                        
                except json.JSONDecodeError as e:
                    print(f"\n⚠️  JSON decode error: {e}")
                    print(f"Raw data: {data_str}")
        
        # Print summary
        print("\n" + "="*50)
        print("📋 SUMMARY")
        print("="*50)
        print(f"✅ Response completed successfully")
        print(f"📏 Answer length: {len(full_answer)} characters")
        print(f"⏱️  Response time: {metadata.get('response_time', 0):.2f}s")
        print(f"🎯 Context used: {metadata.get('context_used', False)}")
        print(f"🔧 Method: {metadata.get('method', 'unknown')}")
        
        if metadata.get('context_used'):
            print(f"📊 Sources: {metadata.get('num_sources', 0)}")
        else:
            print(f"💡 Used general knowledge (no document context)")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_get_endpoint():
    """Test the GET streaming endpoint"""
    
    question = "What is artificial intelligence?"
    url = f"http://localhost:8000/api/v1/rag/ask-stream?q={requests.utils.quote(question)}&n_results=3&temperature=0.5"
    
    print(f"\n🤖 Testing GET endpoint")
    print("="*50)
    print(f"Question: {question}")
    print("\n🔄 Streaming response:")
    print("-"*30)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return
        
        content_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]
                
                if data_str == "[DONE]":
                    print(f"\n\n✅ GET stream completed! (received {content_count} content chunks)")
                    break
                
                try:
                    chunk = json.loads(data_str)
                    if chunk["type"] == "content":
                        print(chunk["content"], end="", flush=True)
                        content_count += 1
                except json.JSONDecodeError:
                    pass
                    
    except requests.exceptions.RequestException as e:
        print(f"❌ GET request failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Streaming RAG API Tests")
    print("="*60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)
    
    # Test POST endpoint with philosophical question
    test_streaming_with_general_knowledge()
    
    # Wait between tests
    time.sleep(2)
    
    # Test GET endpoint with AI question
    test_get_endpoint()
    
    print("\n" + "="*60)
    print("🎉 All tests completed!")
    print("\n💡 Tips:")
    print("- For document-specific questions, upload documents first using /api/v1/documents/upload")
    print("- The system automatically uses general knowledge for questions without relevant context")
    print("- Access full API docs at: http://localhost:8000/docs")