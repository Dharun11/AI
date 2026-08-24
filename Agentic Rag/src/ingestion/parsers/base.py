from pathlib import Path
from typing import Protocol, runtime_checkable

from ingestion.models import Document, Element


@runtime_checkable
class Parser(Protocol):
    name: str

    def parse(self, source: Path) -> tuple[Document, list[Element]]: ...
