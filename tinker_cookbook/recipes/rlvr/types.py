"""Shared types for RLVR recipes."""

from dataclasses import dataclass
from typing import Callable, Literal

GradeStatus = Literal["correct", "incorrect", "timeout", "error", "ambiguous"]


@dataclass(frozen=True)
class RLVRExample:
    question: str
    reference: str


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    status: GradeStatus
    detail: str = ""


ExtractFn = Callable[[str], str]  # raises ValueError on failure
