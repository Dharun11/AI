import os
from datetime import UTC, datetime
from typing import Any

from ingestion.hashing.factory import register_hash_store
from ingestion.models import Chunk

DEFAULT_BATCH_SIZE = 500

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS chunk_hashes (
    project_id   VARCHAR(128) NOT NULL,
    content_hash CHAR(64)     NOT NULL,
    doc_id       VARCHAR(128) NOT NULL,
    chunk_id     VARCHAR(255) NOT NULL,
    embedded_at  DATETIME(6)  NOT NULL,
    PRIMARY KEY (project_id, content_hash),
    KEY idx_chunk_hashes_project_doc (project_id, doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def _env_or(value: Any, env_var: str, default: Any = None) -> Any:
    return value if value is not None else os.environ.get(env_var, default)


@register_hash_store("mysql")
class MySQLHashStore:
    """MySQL-backed idempotency index shared across many projects.

    SQLiteHashStore gives each project its own file - the equivalent
    isolation here is a `project_id` column that's part of the primary key
    (`PRIMARY KEY (project_id, content_hash)`), so the same content can be
    independently "unseen" for two different projects, enforced by the
    database rather than by which file happened to be opened. One shared
    table also means this survives the API running as more than one
    process/container, unlike a local SQLite file.

    One instance is scoped to a single project (mirrors SQLiteHashStore's
    shape exactly, so IngestionPipeline needs no changes), but many instances
    are meant to share one connection pool - construct the pool once (see
    `build_pool`) and pass it to every per-project instance, rather than
    letting each instance open its own connections.
    """

    name = "mysql"

    def __init__(
        self,
        project_id: str,
        pool: Any = None,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._project_id = project_id
        self._batch_size = batch_size
        self._pool = pool or self.build_pool(
            host=host, port=port, user=user, password=password, database=database
        )
        # Schema is verified once per pool (not once per project/request) -
        # a plain attribute on the pool object is enough of a guard, since a
        # pool is already the thing shared across every per-project instance.
        if not getattr(self._pool, "_chunk_hashes_ready", False):
            self._ensure_table()
            self._pool._chunk_hashes_ready = True

    @staticmethod
    def build_pool(
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        mincached: int = 1,
        maxcached: int = 5,
    ) -> Any:
        # Lazy imports: importing this module for registration must not force
        # pymysql/DBUtils to be installed for anyone not using the mysql
        # hash store, same discipline as qdrant_client/sentence_transformers.
        import pymysql
        from dbutils.pooled_db import PooledDB

        host = _env_or(host, "MYSQL_HOST")
        port = int(_env_or(port, "MYSQL_PORT", 3306))
        user = _env_or(user, "MYSQL_USER")
        password = _env_or(password, "MYSQL_PASSWORD", "")
        database = _env_or(database, "MYSQL_DATABASE")

        if not host or not user or not database:
            raise ValueError(
                "MySQLHashStore requires host/user/database - pass them explicitly or set "
                "MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE "
                "(MYSQL_PORT defaults to 3306)."
            )

        return PooledDB(
            pymysql,
            mincached=mincached,
            maxcached=maxcached,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            autocommit=True,
            charset="utf8mb4",
        )

    def _ensure_table(self) -> None:
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(_TABLE_DDL)
        finally:
            conn.close()

    def contains(self, content_hash: str) -> bool:
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM chunk_hashes WHERE project_id = %s AND content_hash = %s",
                    (self._project_id, content_hash),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()

    def get_record(self, content_hash: str) -> dict[str, str] | None:
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, chunk_id, embedded_at FROM chunk_hashes "
                    "WHERE project_id = %s AND content_hash = %s",
                    (self._project_id, content_hash),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"doc_id": row[0], "chunk_id": row[1], "embedded_at": str(row[2])}

    def filter_unseen(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        seen: set[str] = set()
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                for start in range(0, len(chunks), self._batch_size):
                    batch = chunks[start : start + self._batch_size]
                    hashes = [chunk.content_hash for chunk in batch]
                    placeholders = ",".join(["%s"] * len(hashes))
                    cur.execute(
                        f"SELECT content_hash FROM chunk_hashes "
                        f"WHERE project_id = %s AND content_hash IN ({placeholders})",
                        [self._project_id, *hashes],
                    )
                    seen.update(row[0] for row in cur.fetchall())
        finally:
            conn.close()

        return [chunk for chunk in chunks if chunk.content_hash not in seen]

    def mark_embedded(self, chunk: Chunk) -> None:
        self.mark_all_embedded([chunk])

    def mark_all_embedded(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embedded_at = datetime.now(UTC)
        rows = [
            (self._project_id, chunk.content_hash, chunk.doc_id, chunk.chunk_id, embedded_at)
            for chunk in chunks
        ]
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunk_hashes (project_id, content_hash, doc_id, chunk_id, embedded_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        doc_id = VALUES(doc_id),
                        chunk_id = VALUES(chunk_id),
                        embedded_at = VALUES(embedded_at)
                    """,
                    rows,
                )
        finally:
            conn.close()

    def delete_by_doc_id(self, doc_id: str) -> None:
        conn = self._pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunk_hashes WHERE project_id = %s AND doc_id = %s",
                    (self._project_id, doc_id),
                )
        finally:
            conn.close()
