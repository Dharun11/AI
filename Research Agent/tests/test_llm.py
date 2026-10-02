import pytest

from research_agent import llm as llm_mod
from research_agent.config import get_settings


@pytest.mark.parametrize("provider, cls_name, key_env", [
    ("anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
    ("openai", "ChatOpenAI", "OPENAI_API_KEY"),
    ("deepseek", "ChatDeepSeek", "DEEPSEEK_API_KEY"),
])
def test_factory_selects_provider(monkeypatch, provider, cls_name, key_env):
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.setattr(get_settings(), "llm_provider", provider)
    monkeypatch.setattr(get_settings(), "llm_model", "")
    model = llm_mod.get_llm()
    assert type(model).__name__ == cls_name
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    assert name == llm_mod.DEFAULT_MODELS[provider]


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_provider", "nope")
    with pytest.raises(ValueError):
        llm_mod.get_llm()
