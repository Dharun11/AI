"""Chat model factory. Provider, model id and reasoning level all come from the environment.

    LLM_PROVIDER     anthropic | openai | gemini | deepseek          (required)
    LLM_MODEL        exact model id, no default is built in            (required)
    LLM_REASONING    unset = the model's own default; otherwise off | minimal | low | medium | high | xhigh | max
    LLM_TEMPERATURE  number (default 0) or "none" to leave it out
    LLM_MAX_TOKENS   output cap; unset = the provider's own default (16000 for anthropic, which requires one)
    LLM_EXTRA_PARAMS JSON object passed straight to the model class, for anything not covered above

What each provider means by a reasoning level (checked against the provider docs and the installed
LangChain classes):

    deepseek   off -> thinking disabled. low | high | max -> thinking enabled + reasoning_effort.
               V4 models think by default and reject a forced tool call, so thinking runs use JSON mode.
    anthropic  off -> thinking disabled. low..max -> adaptive thinking + effort.
               Some models cannot turn thinking off (the API answers 400); use LLM_EXTRA_PARAMS then.
    openai     reasoning_effort, with off sent as "none". Some models reject "none" or temperature.
    gemini     off -> thinking_budget=0. minimal | low | medium | high -> thinking_level (Gemini 3+).
               Gemini 2.5 takes a token budget instead: LLM_EXTRA_PARAMS='{"thinking_budget": 1024}'.
"""
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from .config import Settings, get_settings

PROVIDERS = ("anthropic", "openai", "gemini", "deepseek")
OFF = {"off", "none"}
# Only Anthropic requires an output cap. Thinking tokens count toward it, and the SDK refuses non-streaming
# requests above about 21,000, so 16,000 leaves room for thinking. Other providers keep their own default
# (DeepSeek: 8K without thinking, 64K with thinking), which a small cap of ours would silently cut.
DEFAULT_MAX_TOKENS = {"anthropic": 16000}


class LLMConfigError(ValueError):
    """The LLM settings are missing or inconsistent. The message says which environment variable to fix."""


def _require(s: Settings) -> tuple[str, str]:
    provider, model = s.llm_provider.strip().lower(), s.llm_model.strip()
    if not provider:
        raise LLMConfigError(f"LLM_PROVIDER is not set. Set it in .env to one of: {', '.join(PROVIDERS)}.")
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}.")
    if not model:
        raise LLMConfigError(f"LLM_MODEL is not set. Set it in .env to a model id for provider '{provider}'; no default is built in.")
    return provider, model


def reasoning_kwargs(provider: str, level: str) -> dict[str, Any]:
    """Translate the provider-neutral LLM_REASONING level into the constructor arguments for `provider`."""
    level = level.strip().lower()
    if not level:
        return {}                                   # send nothing: the model's own default applies
    off = level in OFF

    def unsupported(allowed: str) -> LLMConfigError:
        return LLMConfigError(f"LLM_REASONING='{level}' is not supported for {provider}. Use: {allowed}.")

    if provider == "deepseek":
        if off:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        if level in ("low", "high", "max"):
            return {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": level}
        raise unsupported("off, low, high, max")
    if provider == "anthropic":
        if off:
            return {"thinking": {"type": "disabled"}}
        if level in ("low", "medium", "high", "xhigh", "max"):
            return {"thinking": {"type": "adaptive"}, "reasoning_effort": level}
        raise unsupported("off, low, medium, high, xhigh, max")
    if provider == "openai":
        if off:
            return {"reasoning_effort": "none"}
        if level in ("minimal", "low", "medium", "high", "xhigh", "max"):
            return {"reasoning_effort": level}
        raise unsupported("off, minimal, low, medium, high, xhigh, max")
    if provider == "gemini":
        if off:
            return {"thinking_budget": 0}
        if level in ("minimal", "low", "medium", "high"):
            return {"thinking_level": level}
        raise unsupported("off, minimal, low, medium, high")
    raise LLMConfigError(f"Unknown provider '{provider}'.")


def _temperature(s: Settings) -> dict[str, float]:
    raw = s.llm_temperature.strip().lower()
    if raw in ("", "none", "off"):
        return {}
    try:
        return {"temperature": float(raw)}
    except ValueError as e:
        raise LLMConfigError(f"LLM_TEMPERATURE='{s.llm_temperature}' must be a number or 'none'.") from e


def get_llm() -> BaseChatModel:
    s = get_settings()
    provider, model = _require(s)
    kwargs: dict[str, Any] = {"model": model, **_temperature(s), **reasoning_kwargs(provider, s.llm_reasoning)}
    max_tokens = s.llm_max_tokens or DEFAULT_MAX_TOKENS.get(provider)
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(s.llm_extra_params)                # explicit escape hatch wins over everything above

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(max_retries=3, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(max_retries=3, **kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(max_retries=3, **kwargs)
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(max_retries=3, **kwargs)


def uses_json_mode(s: Settings | None = None) -> bool:
    """DeepSeek V4 thinks by default and rejects the forced tool call that normal structured output uses.
    So unless thinking is explicitly off, ask DeepSeek for a JSON object and validate it ourselves."""
    s = s or get_settings()
    return s.llm_provider.strip().lower() == "deepseek" and s.llm_reasoning.strip().lower() not in OFF


def structured(llm: BaseChatModel, schema: type[BaseModel], messages: list[BaseMessage]):
    """Return (runnable, messages) that make `llm` answer with a `schema` object, in the way this provider allows."""
    if not uses_json_mode():
        return llm.with_structured_output(schema), messages
    instruction = ("\n\nReply with ONLY a single JSON object (no prose, no code fence) that conforms to this JSON schema:\n"
                   + json.dumps(schema.model_json_schema(), indent=1))
    first = messages[0]
    patched = ([SystemMessage(str(first.content) + instruction)] if isinstance(first, SystemMessage)
               else [SystemMessage(instruction.strip())] + list(messages[:1]))
    return llm.with_structured_output(schema, method="json_mode"), patched + list(messages[1:])
