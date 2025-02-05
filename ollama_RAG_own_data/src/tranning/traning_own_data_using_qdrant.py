import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents.base import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_ollama import OllamaEmbeddings
import glob
import sys


def pdf_file_loader(document_path):
    if document_path:

        pdf_files = glob.glob(document_path + "/*.pdf")

        documents = []
        for pdf_file in pdf_files:
            # Initialize the loader for the current PDF
            loader = UnstructuredPDFLoader(
                pdf_file, mode="single", strategy="fast")

            # Load the PDF content
            docs = loader.load()

            documents.extend(docs)

        return documents
    else:
        print("Upload a PDF file")
        return None


def read_data_in_chunks(data):
    chunks = []
    for doc in data:
        text_spllitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("DOCUMENT_CHUNK_SIZE")), chunk_overlap=int(os.getenv("DOCUMENT_CHUNK_OVERLAP")))
        chunks.extend(text_spllitter.split_documents(data))
    return chunks


def add_data_to_db(chunks):
    # Initialize the Qdrant client
    client = QdrantClient(os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))
    COLLECTION_NAME = os.getenv("POSTGRES_DB_COLLECTION_NAME")

    # Ensure chunks are not empty
    if not chunks:
        print("Chunks list is empty. No data to embed.")
        return

    # Check if the collection exists
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if COLLECTION_NAME not in collection_names:
        # Create the collection if it doesn't exist
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    # Initialize the embedding model
    embedding_model = OllamaEmbeddings(model=os.getenv("QDRANT_EMBEDDING_MODEL"))
    
    if isinstance(chunks[0], Document):  
        chunks = [doc.page_content for doc in chunks]
    
    # Generate embeddings
    embedded_texts = embedding_model.embed_documents(chunks)
    print(f"Generated {len(embedded_texts)} embeddings for {len(chunks)} chunks")

    # Check if embeddings are empty
    if not embedded_texts:
        print("Embeddings are empty. Check the embedding model.")
        return

    # Ensure vector size matches Qdrant collection settings
    if len(embedded_texts[0]) != 1536:
        print(f"Error: Expected vector size 1536 but got {len(embedded_texts[0])}")
        return

    # Insert embeddings into Qdrant
    points = [
        models.PointStruct(id=i, vector=vector, payload={"text": chunk})
        for i, (vector, chunk) in enumerate(zip(embedded_texts, chunks))
    ]

    if not points:
        print("No points to insert. Aborting upsert.")
        return

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Embedded data inserted successfully!")


def main():
    load_dotenv()
    document_path = os.getenv("DOCUMENT_PATH")

    data = pdf_file_loader(document_path)
    chunks = read_data_in_chunks(data)
    add_data_to_db(chunks)


if __name__ == "__main__":
    main()
