"""Inspect AI integration for debate evaluation."""

from .dataset_adapter import DatasetAdapter, GPQAAdapter
from .evaluator import DebateInspectEvaluator, DebateInspectEvaluatorBuilder
from .inspect_task import debate_eval, debate_scorer, debate_solver

__all__ = [
    "DatasetAdapter",
    "DebateInspectEvaluator",
    "DebateInspectEvaluatorBuilder",
    "GPQAAdapter",
    "debate_eval",
    "debate_scorer",
    "debate_solver",
]
