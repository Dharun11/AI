"""Pipeline tracing: one log line per node and per LLM call.

INFO  = what went in / came out, as counts plus a one-line preview.
DEBUG = the full prompt sent to the LLM and its full structured answer.
Every line carries a short run id so concurrent requests can be told apart.
"""
import functools
import logging
import time
from contextvars import ContextVar
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from ..config import get_settings

log = logging.getLogger("research_agent.trace")
run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
# Optional callback that receives short human-readable progress lines (used by the Streamlit UI).
progress_var: ContextVar[Callable[[str], None] | None] = ContextVar("progress", default=None)


def emit(message: str) -> None:
    callback = progress_var.get()
    if callback:
        try:
            callback(message)
        except Exception:  # a broken UI callback must never break the pipeline
            log.debug("progress callback failed", exc_info=True)

_PREVIEW_ATTRS = ("statement", "summary", "facet", "title")


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        return True


def configure_logging() -> None:
    """Idempotent. Our package logs at LOG_LEVEL; third-party libraries (httpx, openai...) stay at WARNING."""
    s = get_settings()
    root = logging.getLogger()
    if not any(getattr(h, "_research_agent", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler._research_agent = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s [%(run_id)s] %(message)s", "%H:%M:%S"))
        handler.addFilter(_RunIdFilter())
        try:  # a Windows console can't print every character the sources contain
            handler.stream.reconfigure(errors="replace")
        except Exception:
            pass
        root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("research_agent").setLevel(s.log_level.upper())


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split()) if limit < 400 else str(text)
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


def _preview(item: Any) -> str:
    if isinstance(item, BaseModel):
        for attr in _PREVIEW_ATTRS:
            if hasattr(item, attr):
                return clip(getattr(item, attr), 90)
    return clip(item, 90)


def brief(value: Any) -> str:
    """Compact one-token description of any state value."""
    if isinstance(value, BaseModel):
        return clip(str(value.model_dump()), 160)
    if isinstance(value, (list, tuple)):
        return f"[{len(value)}]" if not value else f"[{len(value)}] e.g. {_preview(value[0])!r}"
    if isinstance(value, dict):
        return f"{{{len(value)}}}"
    if isinstance(value, str):
        return f"{len(value):,} chars"
    return clip(value, 60)


def size(value: Any) -> str:
    """Even shorter than `brief`: just the shape. Used for node inputs, which are mostly repeats of earlier output."""
    if isinstance(value, (list, tuple, dict)):
        return f"[{len(value)}]"
    if isinstance(value, str):
        return f"={len(value):,}ch"
    return f"<{type(value).__name__}>"


def describe(out: BaseModel | None) -> str:
    """Summarise an LLM structured answer: each list field as a count plus its first item."""
    if out is None:
        return "None (empty answer)"
    parts = []
    for name, value in out:
        parts.append(f"{name}{brief(value)}" if isinstance(value, (list, tuple)) else f"{name}={clip(value, 60)}")
    return "; ".join(parts)


def traced_node(name: str, next_nodes: list[str]):
    """Log a node's input keys, output summary, duration and where control goes next."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            log.info("[node:%s] START  in: %s", name, " ".join(f"{k}{size(v)}" for k, v in state.items()))
            emit(f"[{name}] started")
            t0 = time.perf_counter()
            try:
                out = await fn(state)
            except Exception as e:
                log.error("[node:%s] FAILED after %.1fs: %s: %s", name, time.perf_counter() - t0, type(e).__name__, e)
                raise
            log.info("[node:%s] DONE %.1fs  out: %s  -> %s", name, time.perf_counter() - t0,
                     ", ".join(f"{k}={brief(v)}" for k, v in out.items()), " + ".join(next_nodes))
            emit(f"[{name}] done in {time.perf_counter() - t0:.1f}s")
            return out
        return wrapper
    return deco


class EmptyAnswerError(RuntimeError):
    """The LLM never produced a structured answer, so continuing would silently yield a wrong brief."""


async def ask(llm: BaseChatModel, schema: type[BaseModel], messages: list, label: str,
              accept: Callable[[Any], bool] | None = None, attempts: int = 3):
    """Structured LLM call with request/response logging. All pipeline LLM calls go through here.

    A missing answer (the model replied in prose instead of calling the schema tool) is retried, then raises.
    `accept` lets a caller also reject an answer that parsed but is implausible (e.g. zero groups for 139 claims);
    after the last attempt such an answer is returned as is, with a warning in the log.
    """
    s = get_settings()
    prompt_chars = sum(len(str(m.content)) for m in messages)
    log.info(">> LLM %-18s -> %s  prompt=%s chars", label, schema.__name__, f"{prompt_chars:,}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("   prompt [%s]\n%s", label, clip("\n--- next message ---\n".join(f"[{m.type}] {m.content}" for m in messages), s.log_clip_chars))
    t0 = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            out = await llm.with_structured_output(schema).ainvoke(messages)
        except Exception as e:
            log.error("<< LLM %-18s FAILED after %.1fs: %s: %s", label, time.perf_counter() - t0, type(e).__name__, clip(e, 200))
            raise
        if out is not None and (accept is None or accept(out)):
            break
        log.warning("<< LLM %-18s returned %s (attempt %d/%d)%s", label,
                    "no structured answer" if out is None else "an implausible answer", attempt, attempts,
                    ", retrying" if attempt < attempts else "")
    if out is None:
        raise EmptyAnswerError(f"{label}: the LLM returned no structured answer after {attempts} attempts")
    log.info("<< LLM %-18s %.1fs  %s", label, time.perf_counter() - t0, describe(out))
    emit(f"      LLM {label}: {describe(out)[:90]}")
    if out is not None and log.isEnabledFor(logging.DEBUG):
        log.debug("   answer [%s]\n%s", label, clip(out.model_dump_json(indent=2), s.log_clip_chars))
    return out
