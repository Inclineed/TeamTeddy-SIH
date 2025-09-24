import uvicorn
import sys
import os

# Add the current directory and backend directory to the Python path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")

sys.path.insert(0, current_dir)
sys.path.insert(0, backend_dir)

print(f"Current directory: {current_dir}")
print(f"Backend directory: {backend_dir}")
print(f"Python path: {sys.path[:3]}")

if __name__ == "__main__":
    # Change to backend directory to run the main.py
    os.chdir(backend_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)