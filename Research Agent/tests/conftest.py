from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class _Structured:
    def __init__(self, llm: "FakeLLM", schema):
        self.llm, self.schema = llm, schema

    async def ainvoke(self, messages):
        self.llm.calls.append((self.schema.__name__, messages))
        resp = self.llm.responses[self.schema.__name__]
        return resp(messages) if callable(resp) else resp


class FakeLLM:
    """Stands in for a LangChain chat model: returns canned objects keyed by structured-output schema name."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list = []
        self.structured_kwargs: list = []

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs.append(kwargs)       # e.g. {"method": "json_mode"}
        return _Structured(self, schema)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Tests must not depend on the developer's .env or on a Chromium install: pin the settings and fake the browser."""
    from research_agent.config import get_settings
    from research_agent.fetch import fetcher
    from research_agent.fetch.js import BrowserUnavailable
    s = get_settings()
    for name, value in [("llm_provider", "anthropic"), ("llm_model", "test-model"), ("llm_reasoning", ""),
                        ("llm_temperature", "0"), ("llm_max_tokens", None), ("llm_extra_params", {}),
                        ("use_playwright", True)]:
        monkeypatch.setattr(s, name, value)

    async def no_browser(url, timeout_ms=0, settle_ms=0):
        raise BrowserUnavailable("Chromium is not available in tests")

    monkeypatch.setattr(fetcher, "render_html", no_browser)   # tests that need a browser result override this


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing `text` (Helvetica, one line per \\n)."""
    lines = text.split("\n")
    ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
    for line in lines:
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({esc}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
