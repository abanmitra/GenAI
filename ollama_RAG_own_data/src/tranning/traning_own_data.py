import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_postgres.vectorstores import PGVector
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

            # 1634.78

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


def add_data_to_db(chunk):

    # Connect to PostgreSQL
    connection_string = "postgresql+psycopg2://" + os.getenv("POSTGRES_DB_USER") + ":" + os.getenv("POSTGRES_DB_PASSWORD") + "@" + os.getenv(
        "POSTGRES_DB_HOST") + ":" + os.getenv("POSTGRES_DB_PORT") + "/" + os.getenv("POSTGRES_DB_NAME")

    vector_store = PGVector.from_documents(
        documents=chunk,
        embedding=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
        connection=connection_string,
        collection_name=os.getenv("POSTGRES_DB_COLLECTION_NAME")
    )


def main():
    load_dotenv()
    document_path = os.getenv("DOCUMENT_PATH")

    data = pdf_file_loader(document_path)
    chunks = read_data_in_chunks(data)
    add_data_to_db(chunks)


if __name__ == "__main__":
    main()
