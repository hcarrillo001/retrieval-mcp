"""
Tiny web fetch for URL grounding. Pulls a page and strips it to readable text.

This is intentionally dependency-free (urllib + regex). For bot-protected or
JS-heavy pages, swap in a real extractor (Firecrawl / Bright Data) — same idea,
just a better fetcher.
"""
from __future__ import annotations
import re
import html as _html
import urllib.request


def html_to_text(doc: str) -> str:
    doc = re.sub(r"(?is)<(script|style|noscript|template).*?</\1>", " ", doc)
    doc = re.sub(r"(?s)<[^>]+>", " ", doc)
    doc = _html.unescape(doc)
    return re.sub(r"\s+", " ", doc).strip()


def fetch_text(url: str, limit_chars: int = 12000) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "RetriEval/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode("utf-8", "ignore")
    return html_to_text(raw)[:limit_chars]
