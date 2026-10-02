"""Runtime settings, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
# Settings only reads our own fields; provider SDKs read their API keys (ANTHROPIC_API_KEY,
# DEEPSEEK_API_KEY, ...) from os.environ, so export .env there too. Real env vars take precedence.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "anthropic"
    llm_model: str = ""

    use_playwright: bool = False
    min_text_chars: int = 500          # below this, static extraction is considered a failure
    max_chars_per_source: int = 60_000
    chunk_chars: int = 15_000
    grounding_threshold: int = 85      # rapidfuzz partial_ratio needed to accept a quote
    fetch_timeout: float = 30.0

    output_dir: Path = ROOT / "outputs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
