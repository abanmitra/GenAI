import os

from dotenv import load_dotenv

import chunk_data as chunk
import qdrant_db_injection as qd


def main():
    load_dotenv()
    document_path = os.getenv("DOCUMENT_PATH")

    chunks = chunk.data_chunk(document_path)
    qd.add_data_to_db(chunks)


if __name__ == "__main__":
    main()
