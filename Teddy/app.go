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
		// Try different Python paths
		pythonPaths := []string{
			"../.venv/Scripts/python.exe",
			"python",
			"python3",
		}

		var cmd *exec.Cmd
		for _, pythonPath := range pythonPaths {
			if pythonPath == "../.venv/Scripts/python.exe" {
				// Check if virtual environment exists
				if _, err := os.Stat("../.venv/Scripts/python.exe"); err == nil {
					cmd = exec.Command(pythonPath, "./start_fastapi.py")
					break
				}
			} else {
				// Try system Python with uvicorn
				cmd = exec.Command(pythonPath, "-m", "uvicorn", "backend.python.server:app", "--host", "127.0.0.1", "--port", "8000")
				break
			}
		}

		if cmd == nil {
			fmt.Printf("No suitable Python interpreter found\n")
			return
		}

		cmd.Dir = "." // Set working directory to current directory
		err := cmd.Run()
		if err != nil {
			fmt.Printf("Error starting FastAPI server: %s\n", err)
		}
	}()
}

// CallPython runs hello.py with a parameter
func (a *App) CallPython(name string) string {
	out, err := exec.Command("python", "backend/python/hello.py", name).Output()
	if err != nil {
		return fmt.Sprintf("Error running Python: %s", err)
	}
	return string(out)
}
