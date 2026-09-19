from pydantic import BaseModel

from ingestion.errors import DeadLetterRecord
from ingestion.models import ChunkMetadata


class ChunkingConfigIn(BaseModel):
    strategy: str = "character_v1"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50


class EmbedderConfigIn(BaseModel):
    name: str = "sentence_transformer"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32


class StoreConfigIn(BaseModel):
    name: str = "qdrant"  # "qdrant" | "pinecone" | "in_memory"

    # qdrant
    collection_name: str | None = None
    path: str | None = None
    url: str | None = None

    # pinecone
    pinecone_index_name: str | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"


class IngestRequestConfig(BaseModel):
    # Drives all per-project path/namespace derivation (hash index db,
    # dead-letter dir, default Qdrant path / Pinecone index name) - the one
    # concept this API layer adds on top of the core ingestion library.
    project_id: str
    parser_name: str = "docling"
    chunking: ChunkingConfigIn = ChunkingConfigIn()
    embedder: EmbedderConfigIn = EmbedderConfigIn()
    store: StoreConfigIn = StoreConfigIn()


class ChunkOut(BaseModel):
    chunk_id: str
    content_type: str
    chunking_strategy: str
    token_count: int
    content_hash: str
    text: str
    metadata: ChunkMetadata
    # None when this chunk was an idempotent skip rather than embedded this run.
    embedding_model: str | None = None
    embedding_dim: int | None = None


class DocumentResultOut(BaseModel):
    source_filename: str
    doc_id: str | None
    chunks: list[ChunkOut]
    skipped_count: int
    error: DeadLetterRecord | None


class IngestResponse(BaseModel):
    project_id: str
    counters: dict[str, int]
    documents: list[DocumentResultOut]


class StrategiesResponse(BaseModel):
    parsers: list[str]
    chunkers: list[str]
    embedders: list[str]
    stores: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
