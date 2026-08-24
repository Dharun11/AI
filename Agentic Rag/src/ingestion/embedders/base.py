from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
