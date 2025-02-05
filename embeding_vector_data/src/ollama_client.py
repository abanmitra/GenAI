import ollama
from dotenv import load_dotenv
import os

client = None

def get_embedded_data(document):
    response = client.embeddings(model=os.getenv("OLLAMA_MODEL_NAME"), prompt=document)
    return response["embedding"]

def __init__():
    global client
    # Load environment variable from .env file
    load_dotenv()
    client = ollama.Client()
    # print("\nInitialize the ollama client\n")

# Check if the script is being run directly
if __name__ == "__main__":
    __init__()