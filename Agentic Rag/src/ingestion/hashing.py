import hashlib
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from ingestion.models import Chunk

DEFAULT_BATCH_SIZE = 500


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


class HashIndex:
    """SQLite-backed idempotency index: has this exact chunk content already been embedded?

    Keyed by `content_hash` rather than `chunk_id` — `chunk_id`/`chunk_index`
    can shift between runs (e.g. an earlier chunk being added or removed
    renumbers every chunk after it) even when a given chunk's actual text and
    section context haven't changed at all. `content_hash` is the one field
    that stays stable for genuinely unchanged content across runs, which is
    exactly what "should I skip re-embedding this?" needs to key off of.

    Opens a fresh connection per call rather than holding one open for the
    object's lifetime — this index is used by a batch pipeline, not a hot
    request path, so the extra connect/close cost is negligible and it avoids
    ever holding a stale connection across a long-running ingestion run.
    """

    def __init__(self, db_path: Path | str, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._db_path = str(db_path)
        self._batch_size = batch_size
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_hashes (
                    content_hash TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    embedded_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_hashes_doc_id ON chunk_hashes(doc_id)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def contains(self, content_hash: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM chunk_hashes WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return row is not None

    def get_record(self, content_hash: str) -> dict[str, str] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT doc_id, chunk_id, embedded_at FROM chunk_hashes WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        return {"doc_id": row[0], "chunk_id": row[1], "embedded_at": row[2]}

    def filter_unseen(self, chunks: list[Chunk]) -> list[Chunk]:
        """Returns only the chunks whose content_hash isn't already indexed.

        Looks up hashes in batches (one `IN (...)` query per batch) rather
        than one query per chunk — SQLite also caps how many `?` placeholders
        a single query can take, so very large chunk lists must be batched
        regardless of the round-trip savings.
        """
        if not chunks:
            return []

        seen: set[str] = set()
        with closing(self._connect()) as conn:
            for start in range(0, len(chunks), self._batch_size):
                batch = chunks[start : start + self._batch_size]
                hashes = [chunk.content_hash for chunk in batch]
                placeholders = ",".join("?" * len(hashes))
                rows = conn.execute(
                    f"SELECT content_hash FROM chunk_hashes WHERE content_hash IN ({placeholders})",
                    hashes,
                ).fetchall()
                seen.update(row[0] for row in rows)

        return [chunk for chunk in chunks if chunk.content_hash not in seen]

    def mark_embedded(self, chunk: Chunk) -> None:
        self.mark_all_embedded([chunk])

    def mark_all_embedded(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embedded_at = datetime.now(UTC).isoformat()
        rows = [(chunk.content_hash, chunk.doc_id, chunk.chunk_id, embedded_at) for chunk in chunks]
        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO chunk_hashes (content_hash, doc_id, chunk_id, embedded_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    chunk_id = excluded.chunk_id,
                    embedded_at = excluded.embedded_at
                """,
                rows,
            )
            conn.commit()

    def delete_by_doc_id(self, doc_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM chunk_hashes WHERE doc_id = ?", (doc_id,))
            conn.commit()
