# Multi-Source Research & Synthesis Agent

Give it a **topic and 3–5 URLs** and it returns a Markdown research brief that can be read in about 2 minutes. The brief has five sections:

- **✅ Consensus**: claims that 2+ independent sources agree on.
- **⚔️ Contradictions**: Source A says X, Source B says not-X.
- **🔎 Outliers**: claims made by only one source.
- **🕳️ Gaps**: decision-relevant facets that no source addresses.
- **TL;DR**, plus a sources table and evidence footnotes.

Every claim links to its source URL and has a footnote with the **verbatim quote** that supports it. That quote is checked against the scraped page before the claim can appear in the brief. See [NOTES.md](NOTES.md) for the design and its assumptions.

## How it works
```
fetch ──► extract ──► verify ──┬─► synthesize ─┬─► tldr ──► Markdown
(httpx,   (LLM, per    (quote   │  (LLM groups   │
 trafil.,  chunk,       grounding│   by claim ID, │
 pypdf,    structured)  rapidfuzz)  code-validated)
 playwright)                    └─► gaps ────────┘
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
# optional, for JavaScript-heavy pages:
pip install -e ".[js]" && playwright install chromium

copy .env.example .env          # then set LLM_PROVIDER and the matching API key
```

`LLM_PROVIDER` can be `anthropic` (default model `claude-sonnet-5-5`), `openai` (`gpt-4.1`), `gemini` (`gemini-2.5-pro`) or `deepseek` (`deepseek-flash`; or `deepseek-v4-pro`). The pipeline relies on structured output (tool calling), so pick a model that supports it. Set `LLM_MODEL` to use a different model.

## Run
```bash
uvicorn research_agent.api:app --reload
```

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "Effectiveness of arbitration vs litigation in high-volume consumer disputes in India",
        "urls": [
          "https://example.com/law-journal.pdf",
          "https://example.com/news-article",
          "https://example.com/vendor-blog",
          "https://example.com/high-court-judgment"
        ]
      }'
```

The response body is `text/markdown`. Each brief is also saved to `outputs/<topic-slug>-<timestamp>.md`, and the response's `X-Brief-Path` header gives that path. Interactive docs are at `http://localhost:8000/docs`.

| Endpoint | |
|---|---|
| `POST /research` | `{topic, urls[3..5]}` → Markdown brief. Returns 422 for invalid input, and 502 if fewer than 2 sources were readable. |
| `GET /health` | Liveness check and the active LLM provider. |

## Tests
```bash
pytest
```
The tests use mocked HTTP (`respx`) and a fake LLM, so no API key or network is needed. They cover:
- HTML and PDF parsing, and HTTP errors and paywalls.
- Quote grounding, which rejects fabricated quotes.
- Code-side group validation: single-source "consensus" groups are demoted, and invented claim IDs are dropped.
- API input validation.
- A full end-to-end run that produces a brief.

## Layout
```
src/research_agent/
  api.py               FastAPI app
  config.py, llm.py    settings + provider factory
  models.py            Pydantic schemas (domain + LLM structured output)
  fetch/               httpx fetcher, HTML/PDF/JS extraction, paywall detection
  pipeline/            extract, verify, synthesize, gaps, LangGraph wiring
  render/              Jinja2 Markdown template
tests/
```
