"""HTML -> clean article text. trafilatura first (boilerplate removal), BeautifulSoup fallback."""
import re

import trafilatura
from bs4 import BeautifulSoup

_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg", "button"]


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return soup.title.get_text(strip=True) if soup.title else ""


def bs4_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = [el.get_text(" ", strip=True) for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote"])]
    text = "\n\n".join(b for b in blocks if len(b) > 30)
    return _normalize(text or root.get_text("\n", strip=True))


def extract_html(html: str) -> tuple[str, str]:
    """Return (text, method). Picks the longer of trafilatura / bs4 when trafilatura is thin."""
    traf = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
    traf = _normalize(traf)
    if len(traf) >= 1500:
        return traf, "trafilatura"
    soup_text = bs4_extract(html)
    return (traf, "trafilatura") if len(traf) >= len(soup_text) else (soup_text, "bs4")
