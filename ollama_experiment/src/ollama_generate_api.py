import ollama

# create a client
client = ollama.Client()

# Generate text using the "mistral" model
response = client.generate(
    model="mistral",
    prompt="Why sky is blue",
    stream=True,
)

# Print the generated text
for chunk in response:
    print(chunk["response"], end="", flush=True)
