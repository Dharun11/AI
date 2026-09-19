import os
import time
from typing import Any

from ingestion.models import EmbeddedChunk
from ingestion.store.factory import register_store


@register_store("pinecone")
class PineconeWriter:
    """Pinecone-backed store — a remote managed service, unlike QdrantWriter's
    embedded local mode. It holds no local file lock and needs no close()/
    context-manager discipline, which is exactly why it exists alongside
    Qdrant: many PineconeWriters (one per project) can be constructed and used
    concurrently with zero contention, where opening several embedded Qdrant
    clients against the same path on Windows would deadlock.
    """

    name = "pinecone"

    def __init__(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: str | None = None,
    ) -> None:
        from pinecone import Pinecone

        resolved_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "PineconeWriter requires an api_key (pass it explicitly or set PINECONE_API_KEY)."
            )

        self._pc = Pinecone(api_key=resolved_key)
        self._index_name = index_name
        self._dimension = dimension
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._index: Any = None

    def _index_handle(self) -> Any:
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    def _ensure_index(self) -> None:
        from pinecone import ServerlessSpec

        if self._pc.has_index(self._index_name):
            return
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(cloud=self._cloud, region=self._region),
        )
        # Serverless index creation isn't instant — wait for it to become
        # queryable rather than let the first upsert race a not-ready index.
        for _ in range(30):
            if self._pc.describe_index(self._index_name).status.ready:
                return
            time.sleep(1)

    @staticmethod
    def _chunk_metadata_payload(embedded: EmbeddedChunk) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": embedded.chunk.chunk_id,
            "doc_id": embedded.chunk.doc_id,
            "content_hash": embedded.chunk.content_hash,
            "text": embedded.chunk.text,
            "chunking_strategy": embedded.chunk.chunking_strategy,
            "embedding_model": embedded.embedding_model,
            **embedded.chunk.metadata.model_dump(),
        }
        # Pinecone metadata values must be flat strings/numbers/booleans/
        # string-lists — no nested objects, no None. Qdrant's payload happily
        # stores page_range (a tuple) and a possibly-None doc_title as-is, so
        # this sanitization is the one thing Pinecone needs that Qdrant doesn't.
        page_range = payload.get("page_range")
        if page_range is None:
            payload.pop("page_range", None)
        else:
            # Pinecone only accepts a list of strings, not a list of numbers
            # (confirmed against the real API - a mocked test can't catch this).
            payload["page_range"] = [str(page_range[0]), str(page_range[1])]
        if payload.get("doc_title") is None:
            payload.pop("doc_title", None)
        return payload

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        self._ensure_index()
        vectors = [
            {
                "id": embedded.chunk.chunk_id,
                "values": embedded.vector,
                "metadata": self._chunk_metadata_payload(embedded),
            }
            for embedded in chunks
        ]
        self._index_handle().upsert(vectors=vectors)

    def exists(self, content_hash: str) -> bool:
        if not self._pc.has_index(self._index_name):
            return False
        result = self._index_handle().query(
            vector=[0.0] * self._dimension,
            filter={"content_hash": {"$eq": content_hash}},
            top_k=1,
            include_metadata=False,
        )
        return len(result.matches) > 0

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._pc.has_index(self._index_name):
            return
        self._index_handle().delete(filter={"doc_id": {"$eq": doc_id}})
