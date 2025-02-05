from dotenv import load_dotenv
import os
from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from charset_normalizer import from_path
import sys

# Add the directory containing db_connection to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import db_connection as db


CHUNK_SIZE = None
CHUNK_OVERLAP = None
encoder = None
cursor = None

# The function is used to initialize the environment variables
def init():
    global CHUNK_SIZE, CHUNK_OVERLAP, encoder, cursor

    # Load environment variable from .env file
    load_dotenv()

    CHUNK_SIZE = int(os.getenv("DOCUMENT_CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP"))
    encoder = OllamaDocumentEmbedder(model=os.getenv("OLLAMA_MODEL_NAME"))

    db.__init__()
    cursor = db.get_db()

# The function is used to load documents from a directory
def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                # Detect and normalize encoding
                result = from_path(file_path).best()
                
                if result is None:
                    print(f"Could not detect encoding for {file_path}. Skipping.")
                    continue
                
                # Use .output() to get the text content
                documents.append({"id": filename, "content": result.output()})
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return documents

# The function is used to split a document into chunks
def split_document_into_chunks(content):
    chunks = []
    start = 0
    while start < len(content):
        end = start + CHUNK_SIZE
        chunks.append(content[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

# The function is used to load documents from a directory and split them into chunks
# 1. Load documents from the directory
# 2. Split documents into chunks
def document_load_and_split_into_chunks(directory_path):
    # 1. Load documents from the directory
    documents = load_documents(directory_path)

    # 2. Split documents into chunks and store them in a list
    chunk_documents = []
    for document in documents:
        chunks = split_document_into_chunks(document["content"])
        for i, chunk in enumerate(chunks):
            res = Document.from_dict({
                "content": str(chunk),
                "meta": {
                    "id": f"{document['id']}_chunk{i+1}",
                    "title": document['id'].removesuffix(".txt"),
                }
            })
            # Append the key-value pair dictionary to the list
            chunk_documents.append({
                "original": str(chunk),
                "chunk": res
            })
    return chunk_documents

# The function is used to process documents and store embeddings in the database
def document_embedding_process(directory_path):
    embedded_docs = []
    chunk_documents = document_load_and_split_into_chunks(directory_path)
    for chunk in chunk_documents:
        original = chunk["original"]
        chunk = chunk["chunk"]
        
        result = encoder.run([chunk])
        encode_chunk = result['documents'][0].embedding

        embedded_docs.append({
            "original": original,
            "chunk": encode_chunk
        })
    return embedded_docs

# The function is used to persist the embedding data in the database
def persist_embedding_data(directory_path):
    embedded_docs = document_embedding_process(directory_path)
    for doc in embedded_docs:
        original = doc["original"]
        chunk = doc["chunk"]

        # Insert embedding into the database
        cursor.execute(
            """
            INSERT INTO document_chunk (content, embedding)
            VALUES (%s, %s);
            """,
            (original, chunk),
        )

    # Commit the changes
    cursor.close()
    db.close_db()

def process():
    document_path = "D:/work/GPT/PDF/"
    persist_embedding_data(document_path)

if __name__ == "__main__":
    init()
    process()
