from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import subprocess
import time

app = FastAPI(title="LLM API")

class PromptRequest(BaseModel):  # FIXED: lowercase 'class'
    prompt: str
    model: str = "gemma:2b"  # Changed to gemma
    max_tokens: int = 100

# Start Ollama server if not running
def ensure_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("Ollama server is already running")
            return True
    except:
        print("Starting Ollama server...")
        # Start Ollama in background
        subprocess.Popen(["C:\\Users\\Rush\\AppData\\Local\\Programs\\Ollama\\ollama.exe", "serve"])
        time.sleep(5)  # Wait for server to start
        return True

@app.on_event("startup")
async def startup_event():
    ensure_ollama_running()

@app.get("/")
def read_root():
    return {"message": "LLM API is running", "endpoints": ["/generate", "/models", "/health"]}

@app.post("/generate")
def generate_text(request: PromptRequest):
    """Generate text using Ollama"""
    try:
        # Ollama API endpoint
        ollama_url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        response = requests.post(ollama_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "generated_text": result.get("response", ""),
                "model": result.get("model", request.model),
                "total_duration": result.get("total_duration", 0),
                "prompt_tokens": result.get("prompt_eval_count", 0)
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Ollama server not running. Starting it now..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"models": []}
    except requests.exceptions.ConnectionError:
        return {"models": [], "error": "Ollama server not running. Call /generate first to start it."}
    except Exception as e:
        return {"models": [], "error": str(e)}

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "ollama_running": response.status_code == 200
        }
    except:
        return {"status": "unhealthy", "ollama_running": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")