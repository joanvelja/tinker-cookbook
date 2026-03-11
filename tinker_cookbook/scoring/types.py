"""Core types for binary semantic judging."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Protocol


class BinaryJudgeError(Exception):
    """Base error for binary judge operations."""


class AmbiguousVerdictError(BinaryJudgeError):
    """Raised when a judge response cannot be parsed as positive or negative."""


class BinaryJudgeClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


def normalize_binary_verdict_token(text: str) -> str | None:
    """Normalize a verdict string to its canonical uppercase token.

    Returns None for empty/whitespace input.
    """
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split()[0].rstrip(string.punctuation)
    return token.upper() if token else None


def parse_binary_verdict(
    response: str,
    positive: str,
    negative: str,
    *,
    name: str | None = None,
) -> bool:
    """Parse a binary verdict from raw LLM response text.

    Raises AmbiguousVerdictError if the first token doesn't match positive or negative.
    """
    stripped = response.strip()
    if not stripped:
        raise AmbiguousVerdictError(f"Empty response (template={name!r})")
    token = stripped.split()[0].rstrip(string.punctuation).upper()
    if token == positive.upper():
        return True
    if token == negative.upper():
        return False
    raise AmbiguousVerdictError(
        f"Cannot parse {token!r} as {positive!r} or {negative!r} "
        f"(template={name!r}, full response={response!r})"
    )


@dataclass(frozen=True)
class BinaryJudgeTemplate:
    system: str
    user: str
    positive: str = "CORRECT"
    negative: str = "INCORRECT"
    name: str | None = None

    def render(self, **kwargs: object) -> tuple[str, str]:
        return self.system, self.user.format(**kwargs)

    def parse(self, response: str) -> bool:
        return parse_binary_verdict(response, self.positive, self.negative, name=self.name)
