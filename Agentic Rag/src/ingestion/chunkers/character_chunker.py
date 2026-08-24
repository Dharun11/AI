from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.registry import register_chunker
from ingestion.chunkers.tokens import count_tokens as _count_tokens
from ingestion.chunkers.tokens import decode_tokens, encode_tokens
from ingestion.hashing import sha256_text
from ingestion.models import Chunk, ChunkMetadata, Document, Element, ElementType


def _split_oversized_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Raw token-bounded slicing, used only when one element alone exceeds the budget."""
    tokens = encode_tokens(text)
    if len(tokens) <= max_tokens:
        return [text]

    step = max(max_tokens - overlap_tokens, 1)
    pieces = []
    start = 0
    while start < len(tokens):
        pieces.append(decode_tokens(tokens[start : start + max_tokens]))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return pieces


@register_chunker("character_v1")
class CharacterChunker:
    """Greedily packs elements into token-bounded chunks along natural (element) boundaries.

    Only falls back to raw token slicing when a single element's text alone
    exceeds the chunk size budget — this keeps most chunks aligned to whole
    paragraphs/headings instead of cutting mid-sentence.
    """

    name = "character_v1"

    def chunk(
        self, document: Document, elements: list[Element], config: ChunkerConfig
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer_texts: list[str] = []
        buffer_elements: list[Element] = []
        buffer_tokens = 0
        chunk_index = 0

        def flush() -> None:
            nonlocal buffer_texts, buffer_elements, buffer_tokens, chunk_index
            if not buffer_texts:
                return
            text = "\n\n".join(buffer_texts)
            chunks.append(self._build_chunk(document, buffer_elements, text, chunk_index))
            chunk_index += 1
            buffer_texts, buffer_elements, buffer_tokens = self._seed_overlap(
                buffer_texts, buffer_elements, config.chunk_overlap_tokens
            )

        for element in elements:
            if element.type == ElementType.TABLE or not element.text:
                continue

            element_tokens = _count_tokens(element.text)

            if element_tokens > config.chunk_size_tokens:
                flush()
                for piece in _split_oversized_text(
                    element.text, config.chunk_size_tokens, config.chunk_overlap_tokens
                ):
                    chunks.append(self._build_chunk(document, [element], piece, chunk_index))
                    chunk_index += 1
                continue

            if buffer_texts and buffer_tokens + element_tokens > config.chunk_size_tokens:
                flush()

            buffer_texts.append(element.text)
            buffer_elements.append(element)
            buffer_tokens += element_tokens

        flush()
        return chunks

    @staticmethod
    def _seed_overlap(
        texts: list[str], elements: list[Element], overlap_tokens: int
    ) -> tuple[list[str], list[Element], int]:
        """Carries trailing elements from the flushed chunk into the next one for context continuity.

        A trailing element is only included if it fits within the remaining
        overlap budget — an element larger than `overlap_tokens` is dropped
        rather than included anyway, so the seeded buffer can never itself
        already exceed the overlap budget (which would leave no room for the
        next real element and force an oversized chunk).
        """
        overlap_texts: list[str] = []
        overlap_elements: list[Element] = []
        overlap_count = 0
        for text, element in zip(reversed(texts), reversed(elements)):
            token_count = _count_tokens(text)
            if overlap_count + token_count > overlap_tokens:
                break
            overlap_texts.insert(0, text)
            overlap_elements.insert(0, element)
            overlap_count += token_count
        return overlap_texts, overlap_elements, overlap_count

    @staticmethod
    def _build_chunk(
        document: Document,
        elements: list[Element],
        text: str,
        chunk_index: int,
    ) -> Chunk:
        section_path = elements[0].section_path if elements else []
        page_starts = [e.page_range[0] for e in elements if e.page_range]
        page_ends = [e.page_range[1] for e in elements if e.page_range]
        page_range = (min(page_starts), max(page_ends)) if page_starts else None

        chunk_id = f"{document.doc_id}-chunk-{chunk_index}"
        content_hash = sha256_text(text + "|" + "/".join(section_path))

        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            element_ids=[e.element_id for e in elements],
            text=text,
            content_type="text",
            chunking_strategy=CharacterChunker.name,
            token_count=_count_tokens(text),
            content_hash=content_hash,
            metadata=ChunkMetadata(
                source_uri=document.source_uri,
                doc_title=document.doc_metadata.get("title"),
                section_path=section_path,
                page_range=page_range,
                chunk_index=chunk_index,
                content_type="text",
            ),
        )
