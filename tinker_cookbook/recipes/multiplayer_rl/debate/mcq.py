"""MCQ answer normalization and think-tag stripping."""

from __future__ import annotations

import re

from .parsing import _XML_STRIP_RE

# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think[^>]*>(.*)$", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str | None]:
    """Strip <think> blocks, return (cleaned_text, reasoning)."""
    matches = THINK_RE.findall(text)
    cleaned = THINK_RE.sub("", text).strip()
    if not matches:
        unclosed = _THINK_UNCLOSED_RE.search(text)
        if unclosed:
            matches = [unclosed.group(1)]
            cleaned = text[: unclosed.start()].strip()
        else:
            return text, None
    reasoning = "\n".join(part.strip() for part in matches if part.strip())
    return cleaned, reasoning or None


# ---------------------------------------------------------------------------
# Pre-filter patterns (disqualify ambiguous/hedged responses)
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(
    r"(?i)\b(not sure|uncertain|cannot determine|i(?:'m| am) unsure)\b"
)
_NEGATION_RE = re.compile(r"(?i)\b(?:is not|isn't|cannot be|not)\s+[A-E]\b")
_MULTI_RE = re.compile(
    r"(?i)\b(?:both|either|all of)\s+(?:the above|[A-E]\s+(?:and|or)\s+[A-E])\b"
)
_HEDGE_RANK_RE = re.compile(
    r"(?i)\b(?:most likely|best answer is)\b.*\b(?:but|however|although)\b",
    re.DOTALL,
)

_PRE_FILTERS = [_HEDGE_RE, _NEGATION_RE, _MULTI_RE, _HEDGE_RANK_RE]

# ---------------------------------------------------------------------------
# Cascade extractors (tried in order; first match wins)
# ---------------------------------------------------------------------------

_BARE_MCQ_RE = re.compile(r"^([A-Ea-e])$")
_OPTION_PREFIX_RE = re.compile(r"^\s*\(?([A-Ea-e])\)?\s*[).:\-]")
_ANSWER_FRAME_RE = re.compile(
    r"(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)"
    r"\s*(?:definitely|probably|clearly|obviously|likely|certainly)?"
    r"\s*\(?([A-Ea-e])\)?"
)
_CHOOSE_FRAME_RE = re.compile(
    r"(?i)I\s+(?:choose|pick|select|go with)\s+\(?([A-Ea-e])\)?"
)
_OPTION_LABEL_RE = re.compile(r"(?i)\b(?:option|choice)\s+([A-Ea-e])\b")
_FIRST_LINE_LETTER_RE = re.compile(r"\b([A-E])\b")
_TERMINAL_LETTER_RE = re.compile(r"\b([A-E])\b")


def _is_article_a(letter: str, context: str) -> bool:
    """Check if 'A' at this position is likely an article, not a choice label."""
    if letter != "A":
        return False
    m = re.search(r"\bA\s+(?:lot|few|great|large|small|number|very)\b", context)
    return m is not None


def _extract_bare(cleaned: str) -> str | None:
    m = _BARE_MCQ_RE.match(cleaned)
    return m.group(1).upper() if m else None


def _extract_option_prefix(cleaned: str) -> str | None:
    m = _OPTION_PREFIX_RE.match(cleaned)
    return m.group(1).upper() if m else None


def _extract_answer_frame(cleaned: str) -> str | None:
    m = _ANSWER_FRAME_RE.search(cleaned)
    return m.group(1).upper() if m else None


def _extract_choose_frame(cleaned: str) -> str | None:
    m = _CHOOSE_FRAME_RE.search(cleaned)
    return m.group(1).upper() if m else None


def _extract_option_label(cleaned: str) -> str | None:
    m = _OPTION_LABEL_RE.search(cleaned)
    return m.group(1).upper() if m else None


def _extract_first_line(cleaned: str) -> str | None:
    first_line = cleaned.split("\n")[0].strip()
    if not first_line:
        return None
    matches = _FIRST_LINE_LETTER_RE.findall(first_line)
    candidates = [c for c in matches if not _is_article_a(c, first_line)]
    if len(candidates) == 1:
        return str(candidates[0]).upper()
    return None


def _extract_terminal(cleaned: str) -> str | None:
    all_matches = _TERMINAL_LETTER_RE.findall(cleaned)
    candidates = [c for c in all_matches if not _is_article_a(c, cleaned)]
    if candidates:
        return str(candidates[-1]).upper()
    return None


_EXTRACTORS = [
    _extract_bare,
    _extract_option_prefix,
    _extract_answer_frame,
    _extract_choose_frame,
    _extract_option_label,
    _extract_first_line,
    _extract_terminal,
]


def normalize_mcq(text: str) -> str | None:
    """Extract a single MCQ answer letter (A-E) from free-form text.

    Returns the uppercase letter or ``None`` when extraction is ambiguous
    or the text does not contain a clear MCQ answer (escalation signal).
    """
    cleaned, _ = strip_think(text)
    cleaned = _XML_STRIP_RE.sub("", cleaned).strip()
    if not cleaned:
        return None

    for pattern in _PRE_FILTERS:
        if pattern.search(cleaned):
            return None

    for extractor in _EXTRACTORS:
        result = extractor(cleaned)
        if result is not None:
            return result

    return None
