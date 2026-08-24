from typing import Protocol, runtime_checkable

from ingestion.models import EmbeddedChunk


@runtime_checkable
class VectorStoreWriter(Protocol):
    name: str

    def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    def exists(self, content_hash: str) -> bool: ...

    def delete_by_doc_id(self, doc_id: str) -> None: ...
