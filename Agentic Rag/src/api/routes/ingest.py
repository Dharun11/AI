import tempfile
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from ingestion.models import DocumentResult
from ingestion.observability import RunCounters
from ingestion.pipeline import IngestionPipeline, fold_result_into_counters

from api.dependencies import get_pipeline_builder
from api.schemas import ChunkOut, DocumentResultOut, IngestRequestConfig, IngestResponse

router = APIRouter()


def _save_upload(file: UploadFile, directory: Path) -> Path:
    dest = directory / file.filename
    dest.write_bytes(file.file.read())
    return dest


def _to_document_result_out(result: DocumentResult) -> DocumentResultOut:
    embedded_by_chunk_id = {ec.chunk.chunk_id: ec for ec in result.embedded_chunks}
    chunks_out = [
        ChunkOut(
            chunk_id=chunk.chunk_id,
            content_type=chunk.content_type,
            chunking_strategy=chunk.chunking_strategy,
            token_count=chunk.token_count,
            content_hash=chunk.content_hash,
            text=chunk.text,
            metadata=chunk.metadata,
            embedding_model=embedded_by_chunk_id[chunk.chunk_id].embedding_model
            if chunk.chunk_id in embedded_by_chunk_id
            else None,
            embedding_dim=embedded_by_chunk_id[chunk.chunk_id].embedding_dim
            if chunk.chunk_id in embedded_by_chunk_id
            else None,
        )
        for chunk in result.chunks
    ]
    return DocumentResultOut(
        source_filename=Path(result.source_path).name,
        doc_id=result.document.doc_id if result.document else None,
        chunks=chunks_out,
        skipped_count=result.skipped_count,
        error=result.error,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    files: list[UploadFile] = File(...),
    config: str = Form(...),
    build_pipeline: Callable[[IngestRequestConfig], IngestionPipeline] = Depends(get_pipeline_builder),
) -> IngestResponse:
    # A plain (non-async) def: parsing/embedding are blocking, CPU-bound work,
    # and FastAPI runs sync path operations in a threadpool automatically -
    # an async def here would block the whole event loop for the duration of
    # every ingest call instead.
    try:
        cfg = IngestRequestConfig.model_validate_json(config)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    pipeline = build_pipeline(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        paths = [_save_upload(f, tmp_path) for f in files]
        results = [pipeline.run_one(p) for p in paths]

    counters = RunCounters()
    for result in results:
        fold_result_into_counters(result, counters)

    return IngestResponse(
        project_id=cfg.project_id,
        counters=counters.as_dict(),
        documents=[_to_document_result_out(r) for r in results],
    )
