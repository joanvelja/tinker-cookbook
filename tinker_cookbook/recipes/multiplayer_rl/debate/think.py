"""Canonical think-tag handling for debate recipe.

Pure leaf module — stdlib only, zero debate imports.
"""

import re

THINK_RE = re.compile(r"<think(?:ing)?[^>]*>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think(?:ing)?[^>]*>(.*)$", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str | None]:
    """Strip think tags (closed and unclosed). Returns (cleaned_text, thinking_content)."""
    matches = THINK_RE.findall(text)
    cleaned = THINK_RE.sub("", text).strip()
    if not matches:
        unclosed = _THINK_UNCLOSED_RE.search(text)
        if unclosed:
            matches = [unclosed.group(1)]
            cleaned = text[: unclosed.start()].strip()
        else:
            return text.strip(), None
    reasoning = "\n".join(part.strip() for part in matches if part.strip())
    return cleaned, reasoning or None


def has_think_block(text: str) -> bool:
    """Detect presence of think tags (closed or unclosed)."""
    return bool(THINK_RE.search(text) or _THINK_UNCLOSED_RE.search(text))
