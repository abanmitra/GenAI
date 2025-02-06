import os

from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector


def add_data_to_db(chunk):
    # Connect to PostgreSQL
    connection_string = "postgresql+psycopg2://" + os.getenv("POSTGRES_DB_USER") + ":" + os.getenv(
        "POSTGRES_DB_PASSWORD") + "@" + os.getenv(
        "POSTGRES_DB_HOST") + ":" + os.getenv("POSTGRES_DB_PORT") + "/" + os.getenv("POSTGRES_DB_NAME")

    vector_store = PGVector.from_documents(
        documents=chunk,
        embedding=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
        connection=connection_string,
        collection_name=os.getenv("POSTGRES_DB_COLLECTION_NAME")
    )
