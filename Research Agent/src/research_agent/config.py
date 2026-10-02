"""Runtime settings, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
# Settings only reads our own fields; provider SDKs read their API keys (ANTHROPIC_API_KEY,
# DEEPSEEK_API_KEY, ...) from os.environ, so export .env there too. Real env vars take precedence.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    # env_ignore_empty: a blank `LLM_REASONING=` line in .env means "not set", not an empty string.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_ignore_empty=True)

    # Which model to call. Nothing is built in: both must come from the environment.
    llm_provider: str = ""             # anthropic | openai | gemini | deepseek
    llm_model: str = ""                # exact model id for that provider, e.g. the one in your provider's docs

    # Optional tuning, all from the environment too.
    llm_reasoning: str = ""            # unset = model default; off | minimal | low | medium | high | xhigh | max
    llm_temperature: str = "0"         # a number, or "none" to not send it (some reasoning models reject it)
    llm_max_tokens: int | None = None  # unset = provider default (anthropic: 16000)
    llm_extra_params: dict[str, Any] = {}  # JSON object passed straight to the model constructor

    use_playwright: bool = True        # re-render thin or blocked pages in headless Chromium (needs `playwright install chromium`)
    browser_timeout: float = 10.0      # seconds to wait for the page's DOM to load in the browser
    browser_settle: float = 1.5        # extra seconds for client-side scripts to fill the page
    min_text_chars: int = 500          # below this, static extraction is considered a failure
    max_chars_per_source: int = 60_000
    chunk_chars: int = 15_000
    grounding_threshold: int = 85      # rapidfuzz partial_ratio needed to accept a quote
    fetch_timeout: float = 30.0

    log_level: str = "INFO"            # DEBUG also logs every full LLM prompt and answer
    log_clip_chars: int = 3000         # max chars of a prompt/answer printed at DEBUG

    output_dir: Path = ROOT / "outputs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
