"""Chat model factory selected by LLM_PROVIDER (anthropic | openai | gemini | deepseek)."""
from langchain_core.language_models import BaseChatModel

from .config import get_settings

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-flash",
}


def get_llm() -> BaseChatModel:
    s = get_settings()
    provider = s.llm_provider.strip().lower()
    model = s.llm_model.strip() or DEFAULT_MODELS.get(provider, "")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=8192, max_retries=3)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, max_retries=3)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=3)
    if provider == "deepseek":
        # ChatDeepSeek uses forced tool calling for with_structured_output (DeepSeek has no strict
        # json_schema mode). V4 models default to thinking mode, which rejects a forced tool_choice,
        # so thinking is disabled; extraction/classification calls don't need it.
        # Reads DEEPSEEK_API_KEY; set DEEPSEEK_API_BASE to override the endpoint.
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=model, temperature=0, max_tokens=8192, max_retries=3,
                            extra_body={"thinking": {"type": "disabled"}})
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use anthropic | openai | gemini | deepseek)")
