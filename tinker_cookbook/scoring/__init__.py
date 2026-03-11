"""Shared binary judging infrastructure for scoring and grading."""

from tinker_cookbook.scoring.builder import BinaryJudgeBuilder
from tinker_cookbook.scoring.judge import AsyncBinaryJudge, BatchResult, JudgeBatch
from tinker_cookbook.scoring.types import (
    AmbiguousVerdictError,
    BinaryJudgeClient,
    BinaryJudgeError,
    BinaryJudgeTemplate,
    normalize_binary_verdict_token,
    parse_binary_verdict,
)

__all__ = [
    "AmbiguousVerdictError",
    "AsyncBinaryJudge",
    "BatchResult",
    "BinaryJudgeBuilder",
    "BinaryJudgeClient",
    "BinaryJudgeError",
    "BinaryJudgeTemplate",
    "JudgeBatch",
    "normalize_binary_verdict_token",
    "parse_binary_verdict",
]
