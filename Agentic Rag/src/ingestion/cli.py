from pathlib import Path

import typer
from dotenv import load_dotenv

from ingestion.chunkers.base import ChunkerConfig
from ingestion.chunkers.coordinator import ChunkingCoordinator
from ingestion.config import IngestionSettings
from ingestion.embedders.factory import get_embedder
from ingestion.errors import DeadLetterSink
from ingestion.hashing import get_hash_store
from ingestion.observability import configure_logging
from ingestion.parsers.factory import get_parser
from ingestion.pipeline import IngestionPipeline
from ingestion.store.factory import get_store

app = typer.Typer()


def build_pipeline(settings: IngestionSettings) -> IngestionPipeline:
    """Composition root: the one place allowed to know which concrete class
    backs each stage — everything else in the pipeline only sees Protocols.
    """
    parser = get_parser(settings.parser_name)
    coordinator = ChunkingCoordinator(default_strategy=settings.chunking.default_strategy)
    embedder = get_embedder(
        settings.embedder.name,
        model_name=settings.embedder.model_name,
        batch_size=settings.embedder.batch_size,
    )

    if settings.store.name == "qdrant":
        store_kwargs: dict[str, str] = {"collection_name": settings.store.collection_name}
        if settings.store.path:
            store_kwargs["path"] = settings.store.path
        if settings.store.url:
            store_kwargs["url"] = settings.store.url
        store = get_store("qdrant", **store_kwargs)
    elif settings.store.name == "pinecone":
        store = get_store(
            "pinecone",
            index_name=settings.store.pinecone_index_name or settings.store.collection_name,
            dimension=embedder.dimension,
            metric=settings.store.pinecone_metric,
            cloud=settings.store.pinecone_cloud,
            region=settings.store.pinecone_region,
        )
    else:
        store = get_store(settings.store.name)

    if settings.hash_store_name == "mysql":
        hash_index = get_hash_store("mysql", project_id=settings.hash_store_project_id)
    else:
        hash_index = get_hash_store("sqlite", db_path=settings.hash_index_path)

    return IngestionPipeline(
        parser=parser,
        coordinator=coordinator,
        chunker_config=ChunkerConfig(
            chunk_size_tokens=settings.chunking.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunking.chunk_overlap_tokens,
        ),
        embedder=embedder,
        store=store,
        hash_index=hash_index,
        dead_letter_sink=DeadLetterSink(settings.dead_letter_dir),
    )


@app.command()
def run(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    config_path: Path | None = typer.Option(None, "--config", help="Path to an ingestion YAML config"),
) -> None:
    load_dotenv()  # PINECONE_API_KEY / MYSQL_* etc. live in .env, never in a committed YAML
    configure_logging()
    settings = IngestionSettings.from_yaml(config_path) if config_path else IngestionSettings()

    paths = sorted(p for p in input_dir.rglob("*") if p.is_file())
    pipeline = build_pipeline(settings)
    counters = pipeline.run_batch(paths)

    typer.echo(counters.as_dict())


if __name__ == "__main__":
    app()
