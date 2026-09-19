from typing import Protocol, runtime_checkable

from ingestion.models import Chunk


@runtime_checkable
class HashStore(Protocol):
    """The idempotency contract IngestionPipeline actually depends on.

    Deliberately minimal - just the two calls pipeline.py makes. A concrete
    backend (SQLiteHashStore, MySQLHashStore, ...) is free to expose extra
    operational methods (contains, delete_by_doc_id, ...) beyond this
    Protocol, same as VectorStoreWriter implementations do.
    """

    def filter_unseen(self, chunks: list[Chunk]) -> list[Chunk]: ...

    def mark_all_embedded(self, chunks: list[Chunk]) -> None: ...
