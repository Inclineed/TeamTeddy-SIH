package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
)

type App struct{}

// NewApp creates a new App instance
func NewApp() *App {
	return &App{}
}

// Startup runs when the app starts
func (a *App) Startup(ctx context.Context) {
	// Start FastAPI backend (non-blocking)
	go func() {
		// Try different Python paths in order of preference
		pythonPaths := []string{
			"C:/Users/Acer/Desktop/TeamTeddy/.venv/Scripts/python.exe", // Absolute path to your venv
			".venv/Scripts/python.exe",                                   // Relative path from Teddy directory
			"python",
			"python3",
		}

		var cmd *exec.Cmd
		var pythonFound string
		
		for _, pythonPath := range pythonPaths {
			if pythonPath == "C:/Users/Acer/Desktop/TeamTeddy/.venv/Scripts/python.exe" || pythonPath == ".venv/Scripts/python.exe" {
				// Check if virtual environment exists
				if _, err := os.Stat(pythonPath); err == nil {
					cmd = exec.Command(pythonPath, "start_fastapi.py")
					pythonFound = pythonPath
					fmt.Printf("✅ Using Python from virtual environment: %s\n", pythonPath)
					break
				} else {
					fmt.Printf("❌ Virtual environment not found at: %s\n", pythonPath)
				}
			} else {
				// Try system Python
				testCmd := exec.Command(pythonPath, "--version")
				if err := testCmd.Run(); err == nil {
					cmd = exec.Command(pythonPath, "start_fastapi.py")
					pythonFound = pythonPath
					fmt.Printf("✅ Using system Python: %s\n", pythonPath)
					break
				}
			}
		}

		if cmd == nil {
			fmt.Printf("❌ No suitable Python interpreter found. Please ensure Python is installed and virtual environment is set up.\n")
			return
		}

		cmd.Dir = "." // Set working directory to current directory (Teddy folder)
		
		fmt.Printf("🚀 Starting FastAPI backend with: %s\n", pythonFound)
		fmt.Printf("📁 Working directory: %s\n", cmd.Dir)
		fmt.Printf("🌐 Server will be available at: http://localhost:8000\n")
		
		err := cmd.Run()
		if err != nil {
			fmt.Printf("❌ Error starting FastAPI server: %s\n", err)
		}
	}()
}
