from typing import Any, Dict, List
from haystack import component, Document

@component
class CustomDocWriter:
    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    # The function is used to persist the embedding data in the database
    def __persist_embedding_data(self, documents: List[Document]) -> None:
        for doc in documents:
            self.cursor.execute(
                """
                INSERT INTO document_chunk (content, embedding)
                VALUES (%s, %s);
                """,
                (doc.content, doc.embedding),
            )

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document]) -> Dict[str, int]:
        self.__persist_embedding_data(documents)
        return {"documents_written": len(documents)}