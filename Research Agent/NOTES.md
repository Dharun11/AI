# NOTES: Assumptions, Design Decisions & Trade-offs

Every number below was either read from the code or measured in a real run. Where something is an instruction to the LLM rather than a guarantee, or has only been tried once, the text says so.

## 1. Core principle: no made-up links or quotes
Hallucinated citations are the main failure mode for a legal/PM research tool, so the pipeline is built so the LLM cannot invent a URL or a quote.

- Sources get IDs (`S1`–`S5`) and claims get IDs (`S2-C07`). The LLM never sees or writes a URL. Links, source names and the evidence indicators are added by code from stored data.
- In the grouping, gaps and TL;DR steps the LLM refers to claims by ID, and code checks those IDs.
- Every extracted claim must include a **verbatim supporting quote**. The LLM copies that quote from the page during extraction, and `pipeline/verify.py` checks it against the scraped text: it normalises Unicode, quotes, whitespace and case, then tries an exact substring match, then `rapidfuzz.partial_ratio ≥ 85` (`GROUNDING_THRESHOLD`). A quote shorter than 15 characters is rejected as too short to be evidence. A claim that fails is dropped and counted in the brief's Methodology section.
- **What is not verified:** the claim statement (the LLM's paraphrase of the quote), the group summaries, the one-line position stances and the TL;DR are LLM-written. Verification proves a quote exists on the page, not that the paraphrase is faithful to it.
- Cross-source groups returned by the LLM are **validated in code** (`validate_groups`):
  - IDs that don't exist are dropped.
  - A consensus group must span **≥ 2 distinct sources**. Otherwise it is dissolved, and its claims become single-source claims.
  - A contradiction needs **≥ 2 non-empty positions** that together span **≥ 2 sources**.
  - A claim can sit in at most one consensus group.

## 2. Category definitions
| Category | Definition used |
|---|---|
| **Consensus** | Two or more *different* sources make substantially the same assertion (same direction, compatible numbers). The LLM judges semantic equivalence; code enforces the source rule. |
| **Single-source claim** (called "outlier" in the code) | A verified claim that is in no valid consensus or contradiction group. This is derived in code, not chosen by the LLM. It depends on the LLM's grouping, so a corroborated claim whose match the LLM missed will also appear here. |
| **Contradiction** | Claims from different sources that can't both be true, or that take explicitly opposing positions on the same question. The prompt tells the LLM that differences of scope or emphasis, and overlapping numeric ranges, are *not* contradictions. |
| **Gap** | A **standard decision dimension** that a PM, BA or legal analyst would expect for the topic, but that **no verified claim from any source** addresses. |

### How gaps are found
A gap is something *absent*, so it can't be extracted. It has to be measured against an expectation.

1. **Expected facets.** These are generated from the *topic alone*, before the model sees any source, so they aren't biased by what was found. The prompt gives a fixed baseline: Regulatory/Compliance, Cost/Pricing, Security/Data privacy, Scalability, Enforcement/Legal risk, Timelines, Stakeholder impact, Accessibility/Inclusion. The model is asked to adapt the baseline to the topic and add topic-specific facets (8–12 in total), such as a statute or regulator. The list is regenerated on every run, so its wording changes between runs.
2. **Coverage audit.** The expected facets are mapped against the verified claims. For every facet it calls covered, the auditor must cite the claim IDs that cover it, and code discards citations to IDs that don't exist. A facet with no valid supporting claim is reported as a gap.
3. Gaps have **no citations by design**, because they are the absence of evidence.

## 3. Parsing strategy
| Challenge | Handling |
|---|---|
| Raw HTML, boilerplate | `trafilatura` (main-content extraction) first. If it returns under 1,500 characters, `BeautifulSoup` is also run (it strips nav/script/footer/aside and keeps `article`/`main` blocks) and the longer result is kept. |
| JavaScript-rendered pages | On by default (`USE_PLAYWRIGHT=true`). If a plain download yields fewer than 500 characters (`MIN_TEXT_CHARS`), or the site answers 401/402/403/451, the page is re-opened in headless Chromium: wait up to 10s for `domcontentloaded` (`BROWSER_TIMEOUT`) plus a 1.5s settle (`BROWSER_SETTLE`), then extract again. `networkidle` was used first and dropped, because trackers and streaming requests keep many sites from ever going idle. Measured once each: one OECD page timed out after 30s with no content under `networkidle`, against 1.9s and 236 characters now (still thin, so it shows as `partial`); a WEF report page and a JavaScript demo page gave identical text length in 3.1s and 3.4s instead of 4.6s and 5.4s. |
| Browser visibility and failure | Source status says `processed (browser-rendered)` for sources that needed the browser, and the Streamlit sidebar and `/health` report whether Chromium is installed. If Chromium is missing, crashes or times out, the page keeps its static text and is marked `partial` with a note (for example `could not retry in a browser: Chromium not installed (run: uv run playwright install chromium)`). With no static text at all it is `parse_failed`. A refused page (401/402/403/451) that the browser also cannot get is `blocked`, shown as "inaccessible", with the HTTP code. |
| PDFs (judgments, journals) | Detected by content-type, `.pdf` suffix or `%PDF-` magic bytes. Parsed with `pypdf`, and hyphenated line breaks are re-joined. Scanned or image-only PDFs need OCR, which is out of scope. |
| Refusals, paywalls, timeouts | **Reported, never bypassed.** Statuses: `ok`, `partial`, `blocked` (HTTP 401/402/403/451 or a bot-check page; shown as "inaccessible"), `timeout`, `parse_failed`, `failed` (network error, HTTP 404/5xx). A 403 only proves our fetcher was denied, so it is not called a paywall. That word appears only when the page text says so ("subscribe to continue…", which gives `partial`). The brief's Source status table shows this, so the reader knows what wasn't analysed. |
| Network flakiness | `tenacity` retries network errors and timeouts: 3 attempts with exponential backoff (1–8s). Because timeouts are retried too, a host that never answers can take on the order of 90 seconds before it is reported as `timeout`. URLs are fetched concurrently, so a run waits for the slowest one. |
| Very long documents | Text is capped at `MAX_CHARS_PER_SOURCE` (60,000, with a "truncated" note) and split into 15,000-character chunks, cutting at paragraph boundaries where possible, with 500 characters of overlap. Near-duplicate claims from the same source, such as those created by the overlap, are removed during verification. |

If fewer than **2** sources are readable (`ok` or `partial`), the API returns **502**, because cross-source reasoning is meaningless with one source.

### Extraction coverage
An extraction chunk of 1,000 or more characters that returns no claims is retried once. If it is still empty, the source is marked `partial` with `N/M extraction chunks returned no claims; coverage may be incomplete`, shown in Source status, so a long report that yielded almost nothing is not presented as fully analysed. Shorter chunks (footers, reference lists) may legitimately be empty and are neither retried nor flagged. This was added after a review of a run where two of three chunks of a long report returned nothing without any warning. It is covered by tests; it did not trigger in any of my own runs, and why a model returns an empty list for a relevant chunk is not explained.

## 4. Pipeline & cost
LangGraph: `fetch → extract → verify → (synthesize ∥ gaps) → tldr`, then `render` builds the brief.

LLM calls per run: one per chunk (a source has 1–5 chunks, from the 60,000-character cap and 15,000-character chunks), plus 1 synthesize, 2 gaps and 1 TL;DR. The 5-source example run made 11 calls (7 extract + 4). A retry adds a call.

- The provider, model id and reasoning level are read from the environment (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_REASONING`). No model name is in the code. Temperature defaults to 0.
- Structured output uses `with_structured_output` with Pydantic schemas, and Pydantic validates every response.
- A reply that is missing or unparseable is retried up to 3 times, then raises an error. A long extraction chunk with an empty claims list gets one retry (see above). An empty group list from synthesis gets up to 3 attempts and is then accepted, with a warning in the log.
- DeepSeek V4 models think by default and reject a forced tool call in thinking mode (per DeepSeek's documentation and a call that failed with HTTP 400). So when thinking is not explicitly off, the app asks DeepSeek for a JSON object, adds the schema to the system message, and validates it.
- Reasoning tokens count toward the output cap (documented for Anthropic, and observed with DeepSeek, where a call capped at 8,192 tokens stopped before the JSON was complete). When the SDK reports a length limit, the app raises an error naming `LLM_MAX_TOKENS`. That message has only been seen with DeepSeek; other providers may report truncation differently.
- **Only DeepSeek has been run against a live API.** The Anthropic, OpenAI and Gemini settings follow those providers' documentation and were checked by constructing the LangChain classes and inspecting the request they would send, not by calling the APIs.

### How the brief shows its evidence
Every finding carries an indicator (`Sources: 2 of 5 · Evidence: S1-C02, S5-C01`: plain counts, not a confidence score). A contradiction shows each side as Position A **vs.** Position B, with a one-line stance (asked from the LLM, matched to its position by code, and falling back to the first claim's own text if missing), the sources behind it and its claims. The Evidence chain table lists claim ID → source → exact quote → URL for every claim cited in the brief. URLs and source names come from stored data. The quotes are the LLM-copied text that passed verification.

## 5. Assumptions
- Input is **3–5 unique URLs** and a topic of 3–300 characters. The API and the Streamlit app validate the counts and uniqueness. That the URLs are publicly reachable is an assumption, not a check.
- Sources count as independent. Syndication, such as two news sites republishing the same wire story, is not detected and would inflate consensus.
- Claims are limited to 20 per chunk. Single-source claims are usually the largest category, so the brief shows the first 3 per source in extraction order. The extractor is asked to list the most decision-relevant claims first; that is an instruction and has not been verified. The rest go in a collapsed list, so nothing is hidden from the page.
- The TL;DR prompt tells the LLM to summarise only categories that have findings and never to invent one that is empty. That is an instruction, not a guarantee.
- The brief is Markdown, and the Streamlit app shows the same content with a per-finding evidence expander. Quotes appear in the Evidence chain table and, in the app, in the per-finding expanders. There are no footnotes.

## 6. Known limitations / next steps
- **Results vary between runs (measured).** Same topic and 5 URLs, DeepSeek with reasoning off, 4 consecutive runs: the LLM proposed 17 to 47 groups; consensus items kept ranged 6 to 9 and contradictions kept 2 to 23; only 59–82% of the kept claims were shared between any two runs (the 20-claim cap makes the extractor choose a different 20); and only 2 of the 29 distinct consensus claim-pairs appeared in all four runs. The headline story in the TL;DR stayed similar. With reasoning on, one run each at `low` and `high` gave fewer proposed groups (14 and 10) and none rejected by code, but one run each is not enough to call it more stable. In an earlier batch, 2 of 4 runs returned no groups and the code treated that as "none found"; that case now retries and logs a warning.
- **Correctness of groupings is unmeasured.** There is no labelled set, so precision and recall for consensus and contradictions are unknown.
- **Scale of synthesis:** all claims go into one synthesis call. It has been run with 139 claims (about 25,000 characters of prompt). The ceiling is roughly 5 sources × up to 5 chunks × 20 claims = 500 claims, which has not been tried. Beyond what works, add embedding-based pre-clustering and synthesise per cluster.
- **Source credibility:** a competitor's marketing blog and a High Court judgment currently carry equal weight. A next step is a source-type classifier (primary law, regulator, academic, news, marketing) shown next to every citation.
- **Syndication / duplicate-source detection:** compute near-duplicate scores between sources and treat copies as one voice.
- **Prompt injection:** text from scraped pages is passed to the LLM as data, but there is no defence against a page that tries to instruct it.
- **OCR** for scanned PDFs (e.g. older court orders), using Tesseract or a vision model.
- **Caching:** cache fetches and extractions keyed by URL plus content hash, so re-runs on overlapping URL sets are cheap.
- **Evaluation:** build a small labelled set of topic+URL pairs with expected consensus and contradictions, and track precision/recall per category across prompt and model changes.
- **Environment:** one native crash (exit code 139) was seen once while parsing HTML on Python 3.14.7 and was not reproduced in 300 parsing rounds, so its cause is unknown. The project pins Python 3.12 (`.python-version`) as a precaution.
