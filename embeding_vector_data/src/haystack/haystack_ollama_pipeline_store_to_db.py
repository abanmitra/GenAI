from dotenv import load_dotenv
import os
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from custom_document_writer import CustomDocWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from datetime import datetime, timezone
import sys
from typing import Any

# Add the directory containing db_connection to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import db_connection as db

def init() -> None:
    global CHUNK_SIZE, CHUNK_OVERLAP, encoder, cursor

    # Load environment variable from .env file
    load_dotenv()

    CHUNK_SIZE = int(os.getenv("DOCUMENT_CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP"))
    encoder = OllamaDocumentEmbedder(model=os.getenv("OLLAMA_MODEL_NAME"))

    db.__init__()
    cursor = db.get_db()

def pipeline_process(document_path: str) -> None:
    # Initialize components
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter()
    pdf_converter = PyPDFToDocument()
    writer = CustomDocWriter(cursor=cursor)

    # Create a pipeline
    pipeline = Pipeline()

    # Add components to the pipeline
    pipeline.add_component("encoder", encoder)
    pipeline.add_component("converter", pdf_converter)
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("writer", writer)

    # Connect components in the pipeline
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "encoder")
    pipeline.connect("encoder", "writer")

    # Run the pipeline
    pipeline.run(
        {
            "converter": {
                "sources": [document_path],
                "meta": {"date_added": datetime.now(timezone.utc).isoformat()}
            }
        }
    )

def process() -> None:
    document_path = "C:/Users/aban.m/OneDrive - HCL TECHNOLOGIES LIMITED/work/experiment/GPT/PDF/mallidi-2024-ijca-924268.pdf"
    pipeline_process(document_path)
    cursor.close()
    db.close_db()

if __name__ == "__main__":
    init()
    process()
