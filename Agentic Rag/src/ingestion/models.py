from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    CAPTION = "caption"


class TableData(BaseModel):
    header_rows: list[list[str]]
    body_rows: list[list[str]]
    caption: str | None = None
    markdown: str


class Document(BaseModel):
    doc_id: str
    source_uri: str
    mime_type: str
    content_hash: str
    ingested_at: datetime
    doc_metadata: dict[str, Any] = {}


class Element(BaseModel):
    element_id: str
    doc_id: str
    order_index: int
    type: ElementType
    text: str | None = None
    table_data: TableData | None = None
    page_range: tuple[int, int] | None = None
    section_path: list[str] = []


class ChunkMetadata(BaseModel):
    source_uri: str
    doc_title: str | None = None
    section_path: list[str] = []
    page_range: tuple[int, int] | None = None
    chunk_index: int
    content_type: Literal["text", "table"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    element_ids: list[str]
    text: str
    content_type: Literal["text", "table"]
    chunking_strategy: str
    token_count: int
    content_hash: str
    metadata: ChunkMetadata


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    vector: list[float]
    sparse_vector: dict[int, float] | None = None
    embedding_model: str
    embedding_dim: int
