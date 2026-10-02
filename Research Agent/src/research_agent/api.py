"""FastAPI entrypoint: POST /research -> Markdown research brief."""
import logging
import re
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator

from .config import get_settings
from .pipeline.graph import InsufficientSourcesError, run_research
from .render.markdown import render_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("research_agent")

app = FastAPI(title="Multi-Source Research & Synthesis Agent", version="0.1.0")


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    urls: list[HttpUrl] = Field(min_length=3, max_length=5)

    @field_validator("urls")
    @classmethod
    def unique_urls(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if len({str(u).rstrip("/") for u in v}) != len(v):
            raise ValueError("urls must be unique")
        return v


def _save(topic: str, markdown: str) -> str:
    out_dir = get_settings().output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "brief"
    path = out_dir / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_provider": get_settings().llm_provider}


@app.post("/research", response_class=PlainTextResponse, responses={200: {"content": {"text/markdown": {}}}})
async def research(req: ResearchRequest) -> PlainTextResponse:
    urls = [str(u) for u in req.urls]
    log.info("research topic=%r urls=%s", req.topic, urls)
    try:
        brief = await run_research(req.topic, urls)
    except InsufficientSourcesError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    markdown = render_markdown(brief)
    path = _save(req.topic, markdown)
    log.info("brief saved to %s", path)
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8", headers={"X-Brief-Path": path})
