from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.dependencies import close_cached_qdrant_writers
from api.routes import ingest, meta

# Loaded at import time (before any request is served, and before the
# per-request-cached embedder/store/hash-store getters in dependencies.py
# ever run) - PINECONE_API_KEY / MYSQL_* live in .env, never in a committed file.
load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Cached QdrantWriters are shared across the whole process lifetime (see
    # dependencies.py) and are never closed per-request - close them once here
    # so embedded-mode's Windows file lock is released cleanly on shutdown.
    close_cached_qdrant_writers()


app = FastAPI(title="Agentic RAG Ingestion API", lifespan=lifespan)
app.include_router(ingest.router)
app.include_router(meta.router)
