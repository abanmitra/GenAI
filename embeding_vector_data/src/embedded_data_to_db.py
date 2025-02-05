import db_connection as db
import ollama_client as ollama

# Sample document chunks
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
    "Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data and use it to learn for themselves.",
    "Deep learning is a subset of machine learning that uses artificial neural networks to model and high-level abstractions in data.",
    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."
]

def process():
    # PostgreSQL Connection
    cursor = db.get_db()

    # Generate embeddings for the document chunks and store them in a database
    for doc in documents:
        print(f"document : {doc}")
        # Insert embedding into the database
        cursor.execute(
            """
            INSERT INTO document_chunk (content, embedding)
            VALUES (%s, %s);
            """,
            (doc, ollama.get_embedded_data(doc)),
        )

    # Commit the changes
    cursor.close()
    db.close_db()

def main():
    db.__init__()
    ollama.__init__()
    # print("\nInitialize the values\n")

    process()

if __name__ == "__main__":
    main()




