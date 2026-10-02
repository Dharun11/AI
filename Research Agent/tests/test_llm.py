import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent import llm as llm_mod
from research_agent.config import get_settings
from research_agent.llm import LLMConfigError, reasoning_kwargs, structured
from research_agent.models import ClaimsOut

from .conftest import FakeLLM

KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
CLASSES = {"anthropic": "ChatAnthropic", "openai": "ChatOpenAI", "gemini": "ChatGoogleGenerativeAI", "deepseek": "ChatDeepSeek"}


def use(monkeypatch, provider, model="my-model", **settings):
    monkeypatch.setenv(KEYS[provider], "test-key")
    s = get_settings()
    for name, value in [("llm_reasoning", ""), ("llm_temperature", "0"), ("llm_max_tokens", None), ("llm_extra_params", {})]:
        monkeypatch.setattr(s, name, value)         # every call starts from a clean slate
    monkeypatch.setattr(s, "llm_provider", provider)
    monkeypatch.setattr(s, "llm_model", model)
    for name, value in settings.items():
        monkeypatch.setattr(s, name, value)


@pytest.mark.parametrize("provider", CLASSES)
def test_model_name_comes_only_from_the_environment(monkeypatch, provider):
    use(monkeypatch, provider, model="whatever-the-env-says")
    model = llm_mod.get_llm()
    assert type(model).__name__ == CLASSES[provider]
    assert (getattr(model, "model_name", None) or getattr(model, "model", None)) == "whatever-the-env-says"


def test_no_model_name_is_built_in(monkeypatch):
    use(monkeypatch, "deepseek", model="")
    with pytest.raises(LLMConfigError, match="LLM_MODEL is not set"):
        llm_mod.get_llm()
    assert not hasattr(llm_mod, "DEFAULT_MODELS")


def test_missing_or_unknown_provider(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_provider", "")
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER is not set"):
        llm_mod.get_llm()
    monkeypatch.setattr(get_settings(), "llm_provider", "nope")
    with pytest.raises(LLMConfigError, match="Unknown LLM_PROVIDER"):
        llm_mod.get_llm()


@pytest.mark.parametrize("provider, level, expected", [
    ("deepseek", "off", {"extra_body": {"thinking": {"type": "disabled"}}}),
    ("deepseek", "high", {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high"}),
    ("anthropic", "off", {"thinking": {"type": "disabled"}}),
    ("anthropic", "medium", {"thinking": {"type": "adaptive"}, "reasoning_effort": "medium"}),
    ("openai", "off", {"reasoning_effort": "none"}),
    ("openai", "xhigh", {"reasoning_effort": "xhigh"}),
    ("gemini", "off", {"thinking_budget": 0}),
    ("gemini", "low", {"thinking_level": "low"}),
    ("openai", "", {}),            # unset: send nothing, the model's own default applies
])
def test_reasoning_level_maps_to_each_providers_parameters(provider, level, expected):
    assert reasoning_kwargs(provider, level) == expected


@pytest.mark.parametrize("provider, level", [("deepseek", "medium"), ("anthropic", "minimal"), ("gemini", "max")])
def test_unsupported_reasoning_level_is_rejected_with_the_allowed_values(provider, level):
    with pytest.raises(LLMConfigError, match="Use:"):
        reasoning_kwargs(provider, level)


@pytest.mark.parametrize("provider, level", [("deepseek", "high"), ("anthropic", "high"), ("openai", "high"), ("gemini", "low")])
def test_reasoning_reaches_the_constructed_model(monkeypatch, provider, level):
    use(monkeypatch, provider, llm_reasoning=level)
    model = llm_mod.get_llm()
    assert getattr(model, "reasoning_effort", None) == level or getattr(model, "thinking_level", None) == level


def test_temperature_can_be_left_out_and_extra_params_win(monkeypatch):
    use(monkeypatch, "openai", llm_temperature="none")
    assert llm_mod.get_llm().temperature is None
    use(monkeypatch, "openai", llm_temperature="0.3", llm_extra_params={"temperature": 0.9})
    assert llm_mod.get_llm().temperature == 0.9
    use(monkeypatch, "openai", llm_temperature="warm")
    with pytest.raises(LLMConfigError, match="LLM_TEMPERATURE"):
        llm_mod.get_llm()


def test_only_anthropic_gets_a_default_token_cap_and_it_can_be_overridden(monkeypatch):
    use(monkeypatch, "anthropic")
    assert llm_mod.get_llm().max_tokens == 16000
    use(monkeypatch, "anthropic", llm_max_tokens=2000)
    assert llm_mod.get_llm().max_tokens == 2000
    use(monkeypatch, "deepseek")                      # DeepSeek's own default (64K when thinking) is left alone
    assert llm_mod.get_llm().max_tokens is None
    use(monkeypatch, "deepseek", llm_max_tokens=50000)
    assert llm_mod.get_llm().max_tokens == 50000


def test_deepseek_uses_json_mode_unless_thinking_is_off(monkeypatch):
    use(monkeypatch, "deepseek", llm_reasoning="")            # V4 thinks by default
    assert llm_mod.uses_json_mode()
    use(monkeypatch, "deepseek", llm_reasoning="high")
    assert llm_mod.uses_json_mode()
    use(monkeypatch, "deepseek", llm_reasoning="off")
    assert not llm_mod.uses_json_mode()
    use(monkeypatch, "anthropic", llm_reasoning="high")
    assert not llm_mod.uses_json_mode()


def test_json_mode_adds_the_schema_to_the_system_message(monkeypatch):
    use(monkeypatch, "deepseek", llm_reasoning="high")
    fake = FakeLLM({})
    _, messages = structured(fake, ClaimsOut, [SystemMessage("rules"), HumanMessage("text")])
    assert fake.structured_kwargs == [{"method": "json_mode"}]
    assert messages[0].content.startswith("rules") and "JSON schema" in messages[0].content and "statement" in messages[0].content
    assert messages[1].content == "text"

    use(monkeypatch, "deepseek", llm_reasoning="off")
    fake = FakeLLM({})
    _, messages = structured(fake, ClaimsOut, [SystemMessage("rules"), HumanMessage("text")])
    assert fake.structured_kwargs == [{}] and messages[0].content == "rules"
