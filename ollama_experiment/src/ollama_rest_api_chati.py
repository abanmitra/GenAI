import ollama
import json

client = ollama.Client()

response = client.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": "How many planate in the solar system? display in a table"},
    ],
    stream=True,
)

for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)