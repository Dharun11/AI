# NOTES: Assumptions, Design Decisions & Trade-offs

## 1. Core principle: the LLM never writes a citation
Hallucinated citations are the main failure mode for a legal/PM research tool, so we don't let the model produce them at all.

- Sources get IDs (`S1`–`S5`) and claims get IDs (`S2-C07`). The LLM only ever sees and returns **IDs**. URLs and quotes are attached by code at render time.
- Every extracted claim must include a **verbatim supporting quote**. `pipeline/verify.py` checks that quote against the scraped text: it normalises Unicode, quotes, whitespace and case, then tries an exact substring match, then `rapidfuzz.partial_ratio ≥ 85`. If the quote isn't found, the claim is dropped, and the count shows up in the brief's Methodology footer.
- Cross-source groups returned by the LLM are **validated in code** (`validate_groups`):
  - IDs that don't exist are dropped.
  - A consensus group must span **≥ 2 distinct sources**. Otherwise it is dissolved, and its claims become outliers.
  - A contradiction needs **≥ 2 non-empty positions** that span **≥ 2 sources**.
  - A claim can sit in at most one consensus group.

## 2. Category definitions
| Category | Definition used |
|---|---|
| **Consensus** | Two or more *different* sources make substantially the same assertion (same direction, compatible numbers). The LLM judges semantic equivalence; code enforces the source rule. |
| **Outlier** | A grounded claim that is in no valid consensus or contradiction group, i.e. made by exactly one source and not contested. This is derived in code, not chosen by the LLM. |
| **Contradiction** | Claims from different sources that can't both be true, or that take explicitly opposing positions on the same question. Differences of scope or emphasis, and numeric ranges that overlap, are *not* contradictions. |
| **Gap** | A **standard decision dimension** that a PM, BA or legal analyst would expect for the topic, but that **no grounded claim from any source** addresses. |

### How gaps are found
A gap is something *absent*, so it can't be extracted. It has to be measured against an expectation.

1. **Expected facets.** These are generated from the *topic alone*, before the model sees any source, so they aren't biased by what was found. Generation starts from a fixed baseline: Regulatory/Compliance, Cost/Pricing, Security/Data privacy, Scalability, Enforcement/Legal risk, Timelines, Stakeholder impact, Accessibility/Inclusion. The model adapts the baseline to the topic and adds topic-specific facets, such as a statute or regulator.
2. **Coverage audit.** The expected facets are mapped against the grounded claims. For every facet it calls covered, the auditor must cite the claim IDs that cover it, and code discards citations to IDs that don't exist. A facet with no valid supporting claim is reported as a gap, so "covered" has to be backed by evidence just like a citation.
3. Gaps have **no citations by design**, because they are the absence of evidence.

## 3. Parsing strategy
| Challenge | Handling |
|---|---|
| Raw HTML, boilerplate | `trafilatura` (main-content extraction) first. If it returns little text, fall back to `BeautifulSoup`, which strips nav/script/footer/aside and keeps `article`/`main` blocks. |
| JavaScript-rendered pages | If static extraction yields fewer than 500 chars and `USE_PLAYWRIGHT=true`, the page is re-rendered in headless Chromium (`networkidle`) and extracted again. Playwright is an optional extra to keep the base install light. |
| PDFs (judgments, journals) | Detected by content-type, `.pdf` suffix or `%PDF-` magic bytes. Parsed with `pypdf`, and hyphenated line breaks are re-joined. Scanned or image-only PDFs need OCR, which is out of scope. |
| Paywalls / bot walls | **Detected, never bypassed.** HTTP 401/402/403/451, paywall phrases ("subscribe to continue…") and bot-check phrases mark a source as `partial` or `failed`. The brief's Sources table shows this status, so the reader knows what wasn't analysed. |
| Network flakiness | `tenacity` retries (3 attempts, exponential backoff), and URLs are fetched concurrently. |
| Very long documents | Text is capped at `MAX_CHARS_PER_SOURCE` (60k) and split into 15k-char chunks on paragraph boundaries with 500 chars of overlap. Duplicates created by the overlap are removed during verification. |

If fewer than **2** sources are readable, the API returns **502**, because cross-source reasoning is meaningless with one source.

## 4. Pipeline & cost
LangGraph: `fetch → extract → verify → (synthesize ∥ gaps) → tldr`

LLM calls per run: one per chunk (typically 1–4 per source), plus 1 synthesize, 2 gaps and 1 TL;DR. For 4 typical articles that comes to **about 8–12 calls**. All calls use `temperature=0` and structured output (Pydantic schemas through `with_structured_output`), so the provider can be switched with `LLM_PROVIDER` (anthropic / openai / gemini / deepseek). DeepSeek has no strict JSON-schema mode, so `ChatDeepSeek` produces structured output through forced tool calling. DeepSeek V4 models default to thinking mode, which rejects a forced `tool_choice`, so thinking is turned off; extraction and classification don't need it. Pydantic still validates every response.

## 5. Assumptions
- Input is **3–5 unique, publicly reachable URLs** (validated by the API) and a topic of 3–300 chars.
- Sources count as independent. We don't detect syndication, e.g. two news sites republishing the same wire story, which would inflate consensus. See Next steps.
- Claims are limited to 20 per chunk. Outliers are usually the largest category, so the brief shows the first 3 per source; the extractor ranks claims by relevance. The rest go in a collapsed `<details>` list, so nothing is lost but the main brief stays short.
- The TL;DR only summarises categories that have findings. It is told never to invent a gap or contradiction when that category is empty.
- Output is Markdown only. Evidence quotes are GitHub-style footnotes, so every citation can be checked in one click.

## 6. Known limitations / next steps
- **Scale of synthesis:** all claims go into one synthesis call. That holds for about 400 claims (5 sources × 60k chars). Beyond that, add embedding-based pre-clustering and synthesise per cluster.
- **Source credibility:** a competitor's marketing blog and a High Court judgment currently carry equal weight. A next step is to add a source-type classifier (primary law, regulator, academic, news, marketing) and show it next to every citation.
- **Syndication / duplicate-source detection:** compute near-duplicate scores between sources and treat copies as one voice.
- **OCR** for scanned PDFs (e.g. older court orders), using Tesseract or a vision model.
- **Caching:** cache fetches and extractions keyed by URL plus content hash, so re-runs on overlapping URL sets are cheap.
- **Evaluation:** build a small labelled set of topic+URL pairs with expected consensus and contradictions, and track precision/recall per category across prompt and model changes.
