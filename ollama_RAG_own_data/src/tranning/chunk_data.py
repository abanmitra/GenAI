import glob
import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def data_chunk(document_path):
    data = pdf_file_loader(document_path)
    chunks = read_data_in_chunks(data)
    return chunks
