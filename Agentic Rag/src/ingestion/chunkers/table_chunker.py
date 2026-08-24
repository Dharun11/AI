from collections.abc import Iterator

from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.registry import register_chunker
from ingestion.chunkers.tokens import count_tokens
from ingestion.hashing import sha256_text
from ingestion.models import Chunk, ChunkMetadata, Document, Element, ElementType, TableData


def _rows_to_markdown(header_rows: list[list[str]], rows: list[list[str]]) -> str:
    lines = []
    if header_rows:
        header = header_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _split_into_row_groups(
    header_rows: list[list[str]], body_rows: list[list[str]], max_tokens: int
) -> list[list[list[str]]]:
    """Groups body rows so each group's rendered markdown (with the header repeated) fits the budget.

    A single row that alone (with the header) exceeds the budget is kept as
    its own oversized group rather than split mid-row — a table row is not
    meaningfully divisible.
    """
    groups: list[list[list[str]]] = []
    current_group: list[list[str]] = []
    for row in body_rows:
        candidate = current_group + [row]
        if current_group and count_tokens(_rows_to_markdown(header_rows, candidate)) > max_tokens:
            groups.append(current_group)
            current_group = [row]
        else:
            current_group = candidate
    if current_group:
        groups.append(current_group)
    return groups


def _split_table(table_data: TableData, max_tokens: int) -> Iterator[tuple[str, int | None, int | None]]:
    """Yields (markdown_text, part_number, total_parts) — part fields are None when the table fits whole."""
    if count_tokens(table_data.markdown) <= max_tokens:
        yield table_data.markdown, None, None
        return

    groups = _split_into_row_groups(table_data.header_rows, table_data.body_rows, max_tokens)
    total_parts = len(groups)
    for part_number, group in enumerate(groups, start=1):
        yield _rows_to_markdown(table_data.header_rows, group), part_number, total_parts


@register_chunker("table_v1")
class TableChunker:
    """Keeps each table intact as its own chunk(s) instead of letting it be character-split.

    A table that fits the token budget becomes one chunk. A table too large
    for the budget is split into row-groups, repeating the header row(s) in
    every part so each part remains a readable, self-contained table.
    """

    name = "table_v1"

    def chunk(
        self, document: Document, elements: list[Element], config: ChunkerConfig
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for element in elements:
            if element.type != ElementType.TABLE or element.table_data is None:
                continue
            for text, part_number, total_parts in _split_table(
                element.table_data, config.chunk_size_tokens
            ):
                chunks.append(
                    self._build_chunk(document, element, text, index, part_number, total_parts)
                )
                index += 1
        return chunks

    @staticmethod
    def _build_chunk(
        document: Document,
        element: Element,
        text: str,
        index: int,
        part_number: int | None,
        total_parts: int | None,
    ) -> Chunk:
        section_path = list(element.section_path)
        if part_number is not None:
            section_path = [*section_path, f"(part {part_number}/{total_parts})"]

        chunk_id = f"{document.doc_id}-chunk-{index}"
        content_hash = sha256_text(text + "|" + "/".join(section_path))

        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            element_ids=[element.element_id],
            text=text,
            content_type="table",
            chunking_strategy=TableChunker.name,
            token_count=count_tokens(text),
            content_hash=content_hash,
            metadata=ChunkMetadata(
                source_uri=document.source_uri,
                doc_title=document.doc_metadata.get("title"),
                section_path=section_path,
                page_range=element.page_range,
                chunk_index=index,
                content_type="table",
            ),
        )
