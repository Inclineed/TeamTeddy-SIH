# Document Processing with LangChain

This module provides comprehensive document processing and chunking functionality using LangChain, preparing documents for embedding generation and RAG (Retrieval-Augmented Generation) applications.

## Features

- **Multi-format Support**: PDF, DOCX, DOC, and TXT files
- **Flexible Chunking**: Multiple text splitting strategies with configurable parameters
- **FastAPI Integration**: RESTful API for document upload and processing
- **Embedding Preparation**: Ready-to-use output for embedding generation
- **Comprehensive Metadata**: Rich metadata for each chunk including source tracking

## File Structure

```
./python/
├── document_processor.py    # Core document processing and chunking logic
├── document_api.py          # FastAPI endpoints for document operations
├── server.py               # Main FastAPI server with integrated endpoints
├── test_processing.py      # Test script for document processing
├── embedding_prep.py       # Embedding preparation demonstration
├── data/
│   ├── test.pdf           # Sample test file
│   └── uploads/           # Directory for uploaded documents
└── requirements.txt        # Python dependencies
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following additional packages for document processing:
```bash
pip install pypdf docx2txt python-magic unstructured pdfplumber
```

## Quick Start

### 1. Basic Document Processing

```python
from document_processor import DocumentProcessor, ChunkingConfig

# Configure chunking parameters
config = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize processor
processor = DocumentProcessor(config)

# Process a single document
processed_doc = processor.process_file("path/to/document.pdf")

print(f"Created {processed_doc.total_chunks} chunks")
for i, chunk in enumerate(processed_doc.chunks[:3]):
    print(f"Chunk {i}: {chunk.page_content[:100]}...")
```

### 2. Using the FastAPI Server

Start the server:
```bash
cd ./python
uvicorn server:app --reload --port 8000
```

Access the API documentation:
- Main API: http://localhost:8000/docs
- Document Processing API: http://localhost:8000/api/docs/docs

### 3. API Endpoints

#### Upload Document
```bash
curl -X POST "http://localhost:8000/api/docs/upload" \
     -F "file=@document.pdf"
```

#### Process Document
```bash
curl -X POST "http://localhost:8000/api/docs/process?filename=document.pdf&chunk_size=800&chunk_overlap=150"
```

#### Get Document Chunks
```bash
curl "http://localhost:8000/api/docs/chunks/document.pdf?start_index=0&limit=5"
```

#### List All Documents
```bash
curl "http://localhost:8000/api/docs/documents"
```

## Configuration Options

### Chunking Configuration

```python
config = ChunkingConfig(
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Overlap between consecutive chunks
    separator="\n\n",       # Primary separator for splitting
    keep_separator=True,    # Whether to keep separators
    length_function=len,    # Function to calculate text length
    is_separator_regex=False # Whether separator is a regex pattern
)
```

### Text Splitter Types

1. **Recursive Character Text Splitter** (Recommended)
   - Tries multiple separators hierarchically
   - Best for most document types

2. **Character Text Splitter**
   - Simple splitting by character count
   - Good for uniform text

3. **Token Text Splitter**
   - Splits by token count (useful for LLM context limits)
   - Requires tokenizer

## Example Usage

### Test the Implementation

```bash
cd ./python
python test_processing.py
```

### Prepare for Embedding Generation

```bash
python embedding_prep.py
```

This will:
1. Process all documents in the `data/` directory
2. Create chunks ready for embedding
3. Export data in JSON format
4. Generate document index for tracking

### Sample Output Format

```json
{
  "id": "document.pdf_0",
  "text": "This is the content of the first chunk...",
  "metadata": {
    "source_file": "document.pdf",
    "chunk_index": 0,
    "total_chunks": 15,
    "file_type": ".pdf",
    "chunk_size": 847,
    "splitter_type": "recursive"
  }
}
```

## Integration with Embedding Models

### OpenAI Embeddings
```python
import openai
from embedding_prep import EmbeddingPreparator

preparator = EmbeddingPreparator()
embedding_data = preparator.prepare_chunks_for_embeddings(processed_doc)
texts = preparator.get_text_only(embedding_data)

# Generate embeddings
embeddings = openai.Embedding.create(
    input=texts,
    model="text-embedding-ada-002"
)
```

### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

### Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
```

## Advanced Features

### Custom Document Loaders

Add support for new document types by extending the `DocumentProcessor` class:

```python
def _load_custom_format(self, file_path: str) -> List[Document]:
    # Custom loading logic
    pass

# Register the loader
processor.supported_extensions['.custom'] = processor._load_custom_format
```

### Batch Processing

Process multiple documents efficiently:

```python
# Process entire directory
processed_docs = processor.process_directory("/path/to/documents")

# Process multiple files
file_paths = ["doc1.pdf", "doc2.docx", "doc3.txt"]
for file_path in file_paths:
    processed_doc = processor.process_file(file_path)
```

### Export and Analysis

```python
# Export chunks to text file for inspection
export_path = processor.export_chunks_to_text(processed_doc)

# Get chunk previews
previews = processor.get_chunk_preview(processed_doc.chunks, num_chunks=5)

# Analyze chunk statistics
chunk_sizes = [len(chunk.page_content) for chunk in processed_doc.chunks]
avg_size = sum(chunk_sizes) / len(chunk_sizes)
```

## Best Practices

1. **Chunk Size Selection**:
   - 500-1000 characters for general use
   - 1500-2000 for longer context models
   - Consider your embedding model's token limits

2. **Overlap Configuration**:
   - 10-20% of chunk size is typically optimal
   - Higher overlap for better context preservation
   - Lower overlap for more distinct chunks

3. **Document Preprocessing**:
   - Clean text before chunking when necessary
   - Handle special characters and formatting
   - Consider document structure (headers, sections)

4. **Metadata Management**:
   - Include relevant source information
   - Track processing parameters
   - Add custom metadata for specific use cases

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Format Issues**: Check file integrity and format support
3. **Memory Issues**: Use smaller chunk sizes for large documents
4. **Encoding Problems**: Specify correct encoding for text files

### Performance Optimization

1. **Caching**: Cache processed documents for repeated access
2. **Async Processing**: Use async/await for concurrent document processing
3. **Streaming**: Process large documents in streaming mode
4. **Batch Operations**: Group operations for better efficiency

## Next Steps

After document chunking, you can:

1. **Generate Embeddings**: Use the prepared chunks with embedding models
2. **Build Vector Database**: Store embeddings in vector databases (Pinecone, Weaviate, Chroma)
3. **Implement RAG**: Create retrieval-augmented generation systems
4. **Search and Similarity**: Build semantic search functionality
5. **Question Answering**: Develop document-based QA systems

## Contributing

To extend this module:

1. Add new document loaders for additional formats
2. Implement custom text splitters for specific use cases
3. Add preprocessing steps for document cleaning
4. Enhance metadata extraction and management
5. Optimize performance for large-scale processing

## License

This implementation is part of the Team Teddy SIH project and follows the project's licensing terms.