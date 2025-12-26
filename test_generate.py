import requests
import json

url = "http://localhost:8001/generate"
data = {
    "prompt": "What is the capital of France?",
    "max_tokens": 50
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))