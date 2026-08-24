from ingestion.models import EmbeddedChunk
from ingestion.store.factory import register_store


@register_store("in_memory")
class InMemoryVectorStoreWriter:
    """Dict-backed store for tests/dev — no persistence, no real similarity search."""

    name = "in_memory"

    def __init__(self) -> None:
        self._points: dict[str, EmbeddedChunk] = {}

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        for embedded in chunks:
            self._points[embedded.chunk.chunk_id] = embedded

    def exists(self, content_hash: str) -> bool:
        return any(embedded.chunk.content_hash == content_hash for embedded in self._points.values())

    def delete_by_doc_id(self, doc_id: str) -> None:
        stale_ids = [
            chunk_id
            for chunk_id, embedded in self._points.items()
            if embedded.chunk.doc_id == doc_id
        ]
        for chunk_id in stale_ids:
            del self._points[chunk_id]

    def get(self, chunk_id: str) -> EmbeddedChunk | None:
        return self._points.get(chunk_id)

    def count(self) -> int:
        return len(self._points)
