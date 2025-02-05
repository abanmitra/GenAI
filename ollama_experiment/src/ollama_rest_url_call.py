import requests
import json

url = "http://localhost:11434/api/generate"

# Populate payload for ollama
payload = {
    "model": "mistral",
    "prompt": "What are you gona say morning in bengali?",
}

response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decode_ine = line.decode('utf-8')
            data = json.loads(decode_ine)
            response_data = data.get('response', "")
            print(response_data, end="" , flush=True)
    print()
else:
    print(f"Error: {response.status_code}")