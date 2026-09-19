"""Composition root for the API: the only module allowed to resolve registry
names into concrete parser/chunker/embedder/store instances, mirroring
ingestion.cli.build_pipeline but from per-request fields instead of a static
YAML settings object.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.coordinator import ChunkingCoordinator
from ingestion.embedders.base import Embedder
from ingestion.embedders.factory import get_embedder
from ingestion.errors import DeadLetterSink
from ingestion.hashing import HashStore, get_hash_store
from ingestion.parsers.base import Parser
from ingestion.parsers.factory import get_parser
from ingestion.pipeline import IngestionPipeline
from ingestion.store.base import VectorStoreWriter
from ingestion.store.factory import get_store

from api.config import ApiSettings
from api.schemas import IngestRequestConfig

_PINECONE_NAME_RE = re.compile(r"[^a-z0-9-]+")

# QdrantWriter instances are never closed per-request (see _get_qdrant_store_cached);
# main.py's shutdown hook closes every one of these once, at process exit, to
# release the embedded-mode file lock cleanly.
_qdrant_writers: list = []


def get_settings() -> ApiSettings:
    return ApiSettings()


def project_paths(settings: ApiSettings, project_id: str) -> tuple[Path, Path]:
    """Per-project dead-letter dir, plus the sqlite hash db path (unused when
    hash_store_name is "mysql", where isolation instead comes from a
    project_id column - see _build_hash_store).

    This path derivation is the entire fix for cross-project idempotency
    collisions in the sqlite case: HashIndex itself dedupes globally by
    content_hash with no project concept, so two projects ingesting
    byte-identical content must simply never share a HashIndex file. No
    core-library change needed - just a distinct path per project_id.
    """
    root = settings.data_root / project_id
    root.mkdir(parents=True, exist_ok=True)
    return root / "hashes.db", root / "dead_letter"


def _slugify_index_name(value: str, max_length: int = 45) -> str:
    slug = _PINECONE_NAME_RE.sub("-", value.lower()).strip("-")
    return slug[:max_length] or "project"


@lru_cache(maxsize=8)
def _get_parser_cached(name: str) -> Parser:
    return get_parser(name)


@lru_cache(maxsize=16)
def _get_embedder_cached(name: str, model_name: str, batch_size: int) -> Embedder:
    return get_embedder(name, model_name=model_name, batch_size=batch_size)


@lru_cache(maxsize=32)
def _get_qdrant_store_cached(collection_name: str, path: str | None, url: str | None) -> VectorStoreWriter:
    # Opening a second embedded-mode QdrantClient against the same path
    # deadlocks on Windows (a file lock held until .close()), so every
    # (collection_name, path, url) combination must resolve to exactly one
    # long-lived writer shared across requests, not one per request.
    store = get_store("qdrant", collection_name=collection_name, path=path, url=url)
    _qdrant_writers.append(store)
    return store


@lru_cache(maxsize=32)
def _get_pinecone_store_cached(
    index_name: str, dimension: int, metric: str, cloud: str, region: str
) -> VectorStoreWriter:
    # Pinecone holds no local file lock, so this caching is purely about
    # avoiding redundant client construction, not correctness.
    return get_store("pinecone", index_name=index_name, dimension=dimension, metric=metric, cloud=cloud, region=region)


@lru_cache(maxsize=32)
def _get_in_memory_store_cached(project_id: str) -> VectorStoreWriter:
    return get_store("in_memory")


@lru_cache(maxsize=1)
def _get_mysql_pool_cached():
    # One pool for the whole process, shared by every project's MySQLHashStore
    # instance - a real MySQL connection is far more expensive to open than a
    # SQLite file, so this is a correctness-adjacent optimization, not just
    # a nice-to-have (opening a fresh pool per request would exhaust the
    # server's max_connections under any real load).
    from ingestion.hashing.mysql_store import MySQLHashStore

    return MySQLHashStore.build_pool()


def _build_hash_store(cfg: IngestRequestConfig, settings: ApiSettings, hash_path: Path) -> HashStore:
    if settings.hash_store_name == "mysql":
        return get_hash_store("mysql", project_id=cfg.project_id, pool=_get_mysql_pool_cached())
    return get_hash_store("sqlite", db_path=hash_path)


def close_cached_qdrant_writers() -> None:
    for writer in _qdrant_writers:
        writer.close()
    _qdrant_writers.clear()
    _get_qdrant_store_cached.cache_clear()


def _build_store(cfg: IngestRequestConfig, settings: ApiSettings, embedder: Embedder) -> VectorStoreWriter:
    store_cfg = cfg.store

    if store_cfg.name == "qdrant":
        path = store_cfg.path
        url = store_cfg.url
        if not path and not url:
            path = str(settings.data_root / cfg.project_id / "qdrant_storage")
        return _get_qdrant_store_cached(store_cfg.collection_name or "chunks", path, url)

    if store_cfg.name == "pinecone":
        index_name = store_cfg.pinecone_index_name or _slugify_index_name(cfg.project_id)
        return _get_pinecone_store_cached(
            index_name, embedder.dimension, store_cfg.pinecone_metric, store_cfg.pinecone_cloud, store_cfg.pinecone_region
        )

    if store_cfg.name == "in_memory":
        return _get_in_memory_store_cached(cfg.project_id)

    return get_store(store_cfg.name)


def build_pipeline_from_request(cfg: IngestRequestConfig, settings: ApiSettings | None = None) -> IngestionPipeline:
    settings = settings or get_settings()

    parser = _get_parser_cached(cfg.parser_name)
    coordinator = ChunkingCoordinator(default_strategy=cfg.chunking.strategy)
    chunker_config = ChunkerConfig(
        chunk_size_tokens=cfg.chunking.chunk_size_tokens,
        chunk_overlap_tokens=cfg.chunking.chunk_overlap_tokens,
    )
    embedder = _get_embedder_cached(cfg.embedder.name, cfg.embedder.model_name, cfg.embedder.batch_size)
    store = _build_store(cfg, settings, embedder)

    hash_path, dead_letter_path = project_paths(settings, cfg.project_id)
    return IngestionPipeline(
        parser=parser,
        coordinator=coordinator,
        chunker_config=chunker_config,
        embedder=embedder,
        store=store,
        hash_index=_build_hash_store(cfg, settings, hash_path),
        dead_letter_sink=DeadLetterSink(dead_letter_path),
    )


def get_pipeline_builder() -> Callable[[IngestRequestConfig], IngestionPipeline]:
    """Indirection point so tests can override the composition root (e.g. to
    force FakeEmbedder + InMemoryVectorStoreWriter) via FastAPI's
    app.dependency_overrides, without needing build_pipeline_from_request
    itself to take test-only parameters.
    """
    return build_pipeline_from_request
