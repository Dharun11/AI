from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.registry import get_chunker
from ingestion.models import Chunk, Document, Element, ElementType

DEFAULT_ELEMENT_TYPE_OVERRIDES: dict[ElementType, str] = {ElementType.TABLE: "table_v1"}


class ChunkingCoordinator:
    """Routes elements to a chunker by type, then merges results into one ordered chunk sequence.

    Elements are grouped into maximal contiguous runs mapped to the same
    strategy (rather than dispatched one at a time) so a strategy like
    CharacterChunker can still pack multiple consecutive elements into one
    chunk — only interrupted where the strategy actually needs to change,
    e.g. around a table. `chunk_id`/`metadata.chunk_index` are reassigned
    after merging since each chunker numbers its own output starting at 0.
    """

    def __init__(
        self,
        default_strategy: str,
        element_type_overrides: dict[ElementType, str] | None = None,
    ) -> None:
        self.default_strategy = default_strategy
        self.element_type_overrides = element_type_overrides or dict(
            DEFAULT_ELEMENT_TYPE_OVERRIDES
        )

    def chunk_document(
        self, document: Document, elements: list[Element], config: ChunkerConfig
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for strategy_name, segment in self._segment_by_strategy(elements):
            chunker = get_chunker(strategy_name)
            chunks.extend(chunker.chunk(document, segment, config))
        return self._renumber(document, chunks)

    def _strategy_for(self, element: Element) -> str:
        return self.element_type_overrides.get(element.type, self.default_strategy)

    def _segment_by_strategy(
        self, elements: list[Element]
    ) -> list[tuple[str, list[Element]]]:
        segments: list[tuple[str, list[Element]]] = []
        current_strategy: str | None = None
        current_elements: list[Element] = []

        for element in elements:
            strategy_name = self._strategy_for(element)
            if strategy_name != current_strategy:
                if current_elements:
                    segments.append((current_strategy, current_elements))
                current_strategy = strategy_name
                current_elements = []
            current_elements.append(element)

        if current_elements:
            segments.append((current_strategy, current_elements))
        return segments

    @staticmethod
    def _renumber(document: Document, chunks: list[Chunk]) -> list[Chunk]:
        renumbered = []
        for index, chunk in enumerate(chunks):
            updated_metadata = chunk.metadata.model_copy(update={"chunk_index": index})
            renumbered.append(
                chunk.model_copy(
                    update={
                        "chunk_id": f"{document.doc_id}-chunk-{index}",
                        "metadata": updated_metadata,
                    }
                )
            )
        return renumbered
