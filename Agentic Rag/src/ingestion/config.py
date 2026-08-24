from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    default_strategy: str = "character_v1"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50


class EmbedderSettings(BaseModel):
    name: str = "sentence_transformer"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32


class StoreSettings(BaseModel):
    name: str = "qdrant"
    collection_name: str = "chunks"
    path: str | None = "./qdrant_storage"
    url: str | None = None


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    parser_name: str = "docling"
    chunking: ChunkingSettings = ChunkingSettings()
    embedder: EmbedderSettings = EmbedderSettings()
    store: StoreSettings = StoreSettings()
    hash_index_path: str = "./ingestion_hashes.db"
    dead_letter_dir: str = "./dead_letter"

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
