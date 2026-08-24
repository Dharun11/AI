from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ingestion.models import Chunk, Document, Element


class ChunkerConfig(BaseModel):
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(
        self, document: Document, elements: list[Element], config: ChunkerConfig
    ) -> list[Chunk]: ...
