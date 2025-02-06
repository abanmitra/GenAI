import logging
import os

from langchain_core.documents.base import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)


def add_data_to_db(chunks):
    # Initialize the Qdrant client
    client = QdrantClient(os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))
    collection_name = os.getenv("POSTGRES_DB_COLLECTION_NAME")

    # Ensure chunks are not empty
    if not chunks:
        logging.info("Chunks list is empty. No data to embed.")
        return

    # Check if the collection exists
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if collection_name not in collection_names:
        # Create the collection if it doesn't exist
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    # Initialize the embedding model
    embedding_model = OllamaEmbeddings(model=os.getenv("QDRANT_EMBEDDING_MODEL"))

    # Conevert
    if isinstance(chunks[0], Document):
        chunks = [doc.page_content for doc in chunks]

    # Generate embeddings
    embedded_texts = embedding_model.embed_documents(chunks)
    logging.info(f"Generated {len(embedded_texts)} embeddings for {len(chunks)} chunks")

    # Check if embeddings are empty
    if not embedded_texts:
        logging.info("Embeddings are empty. Check the embedding model.")
        return

    # Ensure vector size matches Qdrant collection settings
    if len(embedded_texts[0]) != 1536:
        logging.info(f"Error: Expected vector size 1536 but got {len(embedded_texts[0])}")
        return

    # Insert embeddings into Qdrant
    points = [
        models.PointStruct(id=i, vector=vector, payload={"text": chunk})
        for i, (vector, chunk) in enumerate(zip(embedded_texts, chunks))
    ]

    if not points:
        logging.info("No points to insert. Aborting upsert.")
        return

    client.upsert(collection_name=collection_name, points=points)
    logging.info("Embedded data inserted successfully!")

