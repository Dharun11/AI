import uuid
from pathlib import Path
from typing import Self

from ingestion.models import EmbeddedChunk
from ingestion.store.factory import register_store

# Fixed, arbitrary namespace for deriving point ids — must stay constant so the
# same chunk_id always maps to the same point id across runs/processes.
_POINT_ID_NAMESPACE = uuid.UUID("5c3b7a2e-8f2a-4b8f-9e2a-4a2b7c8d9e0f")


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be an unsigned int or a UUID; our chunk_id (e.g.
    "doc-1-chunk-0") is neither, so derive a deterministic UUIDv5 from it —
    re-upserting the same chunk_id always overwrites the same point.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


@register_store("qdrant")
class QdrantWriter:
    """Qdrant-backed store. Runs embedded (no server) when `path` is given, or
    against a real Qdrant server when `url` is given — same API either way,
    since qdrant-client's embedded "local mode" implements the same client
    interface as talking to a server.
    """

    name = "qdrant"

    def __init__(
        self,
        collection_name: str = "chunks",
        path: str | Path | None = None,
        url: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        if (path is None) == (url is None):
            raise ValueError("QdrantWriter requires exactly one of `path` or `url`.")

        self._client = QdrantClient(path=str(path)) if path is not None else QdrantClient(url=url)
        self._collection_name = collection_name

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        from qdrant_client.models import PointStruct

        if not chunks:
            return

        self._ensure_collection(chunks[0].embedding_dim)
        points = [
            PointStruct(
                id=_point_id(embedded.chunk.chunk_id),
                vector=embedded.vector,
                payload={
                    "chunk_id": embedded.chunk.chunk_id,
                    "doc_id": embedded.chunk.doc_id,
                    "content_hash": embedded.chunk.content_hash,
                    "text": embedded.chunk.text,
                    "chunking_strategy": embedded.chunk.chunking_strategy,
                    "embedding_model": embedded.embedding_model,
                    **embedded.chunk.metadata.model_dump(),
                },
            )
            for embedded in chunks
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def exists(self, content_hash: str) -> bool:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self._collection_name):
            return False
        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
            ),
            limit=1,
        )
        return len(results) > 0

    def delete_by_doc_id(self, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self._collection_name):
            return
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )

    def count(self) -> int:
        if not self._client.collection_exists(self._collection_name):
            return 0
        return self._client.count(collection_name=self._collection_name).count

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
