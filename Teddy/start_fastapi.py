import uvicorn
import sys
import os

# Add the current directory to the Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run("backend.python.server:app", host="127.0.0.1", port=8000, reload=True)