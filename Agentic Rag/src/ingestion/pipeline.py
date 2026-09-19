import traceback
from datetime import UTC, datetime
from pathlib import Path

from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.coordinator import ChunkingCoordinator
from ingestion.embedders.base import Embedder
from ingestion.errors import DeadLetterRecord, DeadLetterSink
from ingestion.hashing import HashStore
from ingestion.models import DocumentResult, EmbeddedChunk
from ingestion.observability import RunCounters, get_logger
from ingestion.parsers.base import Parser
from ingestion.store.base import VectorStoreWriter


def fold_result_into_counters(result: DocumentResult, counters: RunCounters) -> None:
    """Accumulates one document's outcome into a batch-level RunCounters.

    Pulled out as a module-level function (not a method) so callers other
    than run_batch - e.g. an API layer processing uploads one at a time via
    run_one - can report the same aggregate counters without duplicating
    this logic.
    """
    if result.error is not None:
        counters.docs_failed += 1
        return
    counters.docs_processed += 1
    counters.chunks_created += len(result.chunks)
    counters.chunks_skipped_idempotent += result.skipped_count


class IngestionPipeline:
    """Wires one document at a time through parse -> chunk -> dedupe -> embed -> store.

    Every collaborator is injected rather than constructed here, so the
    pipeline can be exercised in tests with fakes (a stub embedder, an
    in-memory store) without touching a real model or database — the actual
    wiring of concrete implementations from config belongs in the CLI, not here.
    """

    def __init__(
        self,
        parser: Parser,
        coordinator: ChunkingCoordinator,
        chunker_config: ChunkerConfig,
        embedder: Embedder,
        store: VectorStoreWriter,
        hash_index: HashStore,
        dead_letter_sink: DeadLetterSink,
    ) -> None:
        self._parser = parser
        self._coordinator = coordinator
        self._chunker_config = chunker_config
        self._embedder = embedder
        self._store = store
        self._hash_index = hash_index
        self._dead_letter_sink = dead_letter_sink
        self._logger = get_logger(component="ingestion_pipeline")

    def run_batch(self, paths: list[Path]) -> RunCounters:
        counters = RunCounters()
        for path in paths:
            result = self.run_one(path)
            fold_result_into_counters(result, counters)
        return counters

    def run_one(self, path: Path) -> DocumentResult:
        # Tracks which stage we're in so a failure can be dead-lettered with an
        # accurate `stage` field, without needing a custom exception subclass
        # per stage just to carry that one piece of information.
        stage = "parse"
        try:
            document, elements = self._parser.parse(path)

            stage = "chunk"
            chunks = self._coordinator.chunk_document(document, elements, self._chunker_config)

            stage = "dedupe"
            unseen_chunks = self._hash_index.filter_unseen(chunks)
            skipped_count = len(chunks) - len(unseen_chunks)

            embedded_chunks: list[EmbeddedChunk] = []
            if unseen_chunks:
                stage = "embed"
                vectors = self._embedder.embed_batch([chunk.text for chunk in unseen_chunks])
                embedded_chunks = [
                    EmbeddedChunk(
                        chunk=chunk,
                        vector=vector,
                        embedding_model=self._embedder.name,
                        embedding_dim=self._embedder.dimension,
                    )
                    for chunk, vector in zip(unseen_chunks, vectors, strict=True)
                ]

                stage = "store"
                self._store.upsert(embedded_chunks)

                stage = "mark_embedded"
                self._hash_index.mark_all_embedded(unseen_chunks)

            self._logger.info(
                "document_ingested",
                doc_id=document.doc_id,
                chunks_created=len(chunks),
                chunks_embedded=len(unseen_chunks),
            )
            return DocumentResult(
                source_path=str(path),
                document=document,
                chunks=chunks,
                embedded_chunks=embedded_chunks,
                skipped_count=skipped_count,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - one bad document must never abort the batch
            source_uri = path.resolve().as_uri()
            record = DeadLetterRecord(
                source_uri=source_uri,
                stage=stage,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                timestamp=datetime.now(UTC),
            )
            self._dead_letter_sink.write(record)
            self._logger.error("document_failed", source_uri=source_uri, stage=stage, error=str(exc))
            return DocumentResult(source_path=str(path), document=None, error=record)
