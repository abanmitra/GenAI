from dotenv import load_dotenv
import os
from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
import sys

# Add the directory containing db_connection to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import db_connection as db

embedder = None
cursor = None

# The function is used to initialize the environment variables
def init():
    global embedder, cursor

    # Load environment variable from .env file
    load_dotenv()

    embedder = OllamaDocumentEmbedder(model=os.getenv("OLLAMA_MODEL_NAME"))

    db.__init__()
    cursor = db.get_db()

# The function is used to query data from the database
def query_data_from_db(query):
    query_doc = Document(content=query)
    result = embedder.run([query_doc])
    query_embedding = result['documents'][0].embedding

    print(f"Embedding Query_: {query_embedding}")
    cursor.execute(
        """
        SELECT id, content, 1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM document_chunk
        ORDER BY cosine_similarity DESC LIMIT 5
        """,
        (query_embedding,)
    )
    try:
        print("\nMost similar sentences:")
        for row in cursor.fetchall():
            print(f"\n ID: {row[0]}, CONTENT: {row[1]}, Cosine Similarity: {row[2]} \n")
    except Exception as e:
        print("Error executing query", str(e))
    finally:
        cursor.close()
        db.close_db()

def process():
    query = "What is STORY POINT ESTIMATION MODEL ?"
    query_data_from_db(query)

if __name__ == "__main__":
    init()
    process()
