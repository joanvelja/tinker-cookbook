from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import tinker

if TYPE_CHECKING:
    from tinker_cookbook.usage import UsageTracker

# Set up logger
logger = logging.getLogger(__name__)


class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient
    """

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a TokenCompleter
    """

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        usage_tracker: UsageTracker | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
