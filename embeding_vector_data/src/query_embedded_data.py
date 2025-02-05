import ollama
import db_connection as db
import ollama_client as ollama

def process():
    # PostgreSQL Connection
    cursor = db.get_db()

    query = input("Enter your query: ")
    query_embedding = ollama.get_embedded_data(query)
    # print(f"Embedding Query_: {query_embedding}")

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
        cursor.close()
        db.close_db()

def main():
    db.__init__()
    ollama.__init__()
    # print("\nInitialize the values\n")

    process()

if __name__ == "__main__":
    main()