"""Canonical think-tag handling for debate recipe.

Pure leaf module — stdlib only, zero debate imports.
"""

import functools
import re


@functools.lru_cache(maxsize=16)
def _think_re(tag: str) -> tuple[re.Pattern, re.Pattern]:
    """Compile regex pair for a given tag name. LRU-cached.

    ``think`` and ``thinking`` are treated as aliases — either tag matches
    both ``<think>`` and ``<thinking>`` so models can use either form.
    """
    if tag in ("think", "thinking"):
        pattern = r"think(?:ing)?"
    else:
        pattern = re.escape(tag)
    closed = re.compile(rf"<{pattern}[^>]*>(.*?)</{pattern}>", re.DOTALL | re.IGNORECASE)
    unclosed = re.compile(rf"<{pattern}[^>]*>(.*)$", re.DOTALL | re.IGNORECASE)
    return closed, unclosed


# Precompiled defaults for backward compat (matches <think> OR <thinking>).
THINK_RE = re.compile(r"<think(?:ing)?[^>]*>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think(?:ing)?[^>]*>(.*)$", re.DOTALL | re.IGNORECASE)


def strip_think(text: str, *, tag: str = "thinking") -> tuple[str, str | None]:
    """Strip think tags (closed and unclosed). Returns (cleaned_text, thinking_content)."""
    closed_re, unclosed_re = _think_re(tag)
    matches = closed_re.findall(text)
    cleaned = closed_re.sub("", text).strip()
    if not matches:
        unclosed = unclosed_re.search(text)
        if unclosed:
            matches = [unclosed.group(1)]
            cleaned = text[: unclosed.start()].strip()
        else:
            return text.strip(), None
    reasoning = "\n".join(part.strip() for part in matches if part.strip())
    return cleaned, reasoning or None


def has_think_block(text: str, *, tag: str = "thinking") -> bool:
    """Detect presence of think tags (closed or unclosed)."""
    closed_re, unclosed_re = _think_re(tag)
    return bool(closed_re.search(text) or unclosed_re.search(text))
