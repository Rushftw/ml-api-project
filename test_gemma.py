import requests
import json

print("Testing LLM API with Gemma...")

# Test health
health = requests.get("http://localhost:8001/health")
print(f"Health: {health.json()}")

# Test models
models = requests.get("http://localhost:8001/models")
print(f"Models: {models.json()}")

# Test generation
data = {
    "prompt": "What is the capital of France?",
    "model": "gemma:2b",
    "max_tokens": 50
}

response = requests.post("http://localhost:8001/generate", json=data)
print("\nGeneration Test:")
print(json.dumps(response.json(), indent=2))