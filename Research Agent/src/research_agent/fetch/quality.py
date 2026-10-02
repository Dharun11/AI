"""Heuristics for paywalled / blocked / thin pages. We detect paywalls; we never bypass them."""
import re

_PAYWALL_PATTERNS = re.compile(
    r"subscribe to (continue|read)|to continue reading|this (article|content) is for (subscribers|members)"
    r"|already a subscriber|sign in to (continue|read)|premium (article|content)|unlock this (story|article)",
    re.IGNORECASE,
)
_BLOCK_PATTERNS = re.compile(
    r"access denied|verify you are (a )?human|enable javascript and cookies|captcha|attention required",
    re.IGNORECASE,
)


def assess(text: str, min_chars: int) -> tuple[str, str | None]:
    """Return (status, reason) where status is ok | partial | blocked | parse_failed."""
    if not text.strip():
        return "parse_failed", "no extractable text"
    if _BLOCK_PATTERNS.search(text[:3000]) and len(text) < min_chars * 4:
        return "blocked", "bot protection page instead of the article"
    if _PAYWALL_PATTERNS.search(text):
        return "partial", "paywall detected; only the free portion was analysed"
    if len(text) < min_chars:
        return "partial", f"very little text extracted ({len(text)} chars)"
    return "ok", None
