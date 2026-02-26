"""Evaluator that runs debate evals via Inspect AI."""

from __future__ import annotations

import logging
import os

import chz
import tinker
from inspect_ai import eval_async

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from ..types import ProtocolKind
from .dataset_adapter import DatasetAdapter
from .inspect_task import debate_eval

logger = logging.getLogger(__name__)


@chz.chz
class DebateInspectEvaluatorBuilder:
    """Config for debate evaluation via Inspect AI."""

    adapter: DatasetAdapter
    prompts_ref: str = "scientific_mcq"
    num_rounds: int = 2
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL

    opponent_model: str | None = None  # None = self-play
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    opponent_max_tokens: int = 1024
    judge_max_tokens: int = 512
    renderer_name: str | None = None
    model_name: str | None = None

    log_dir: str | None = None
    log_evals_every: int = 1
    limit: int | None = None
    base_url: str | None = None

    def __call__(self) -> DebateInspectEvaluator:
        return DebateInspectEvaluator(self)


class DebateInspectEvaluator(SamplingClientEvaluator):
    """Runs debate evaluation, optionally writing .eval files via Inspect."""

    def __init__(self, config: DebateInspectEvaluatorBuilder) -> None:
        self._config = config
        self._call_count = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        cfg = self._config
        self._call_count += 1

        if cfg.model_name is None:
            raise ValueError("model_name must be set on DebateInspectEvaluatorBuilder")
        if cfg.renderer_name is None:
            raise ValueError("renderer_name must be set on DebateInspectEvaluatorBuilder")

        renderer = get_renderer(cfg.renderer_name, get_tokenizer(cfg.model_name))

        # Trained model completer (debater A).
        trained_completer = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=cfg.opponent_max_tokens,
            actor="trained",
            model_name=cfg.model_name,
        )

        # Opponent completer (debater B). Self-play if opponent_model is None.
        if cfg.opponent_model is None:
            opponent_completer = TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=cfg.opponent_max_tokens,
                actor="opponent",
                model_name=cfg.model_name,
            )
        else:
            service = tinker.ServiceClient(base_url=cfg.base_url)
            opp_client = service.create_sampling_client(base_model=cfg.opponent_model)
            opp_renderer = get_renderer(cfg.renderer_name, get_tokenizer(cfg.opponent_model))
            opponent_completer = TinkerMessageCompleter(
                sampling_client=opp_client,
                renderer=opp_renderer,
                max_tokens=cfg.opponent_max_tokens,
                actor="opponent",
                model_name=cfg.opponent_model,
            )

        # Judge completer.
        # NOTE: judge_model must be Tinker-compatible for now. External API
        # routing (e.g. "openai/gpt-4o" -> OpenAI direct) is a future concern.
        service = tinker.ServiceClient(base_url=cfg.base_url)
        judge_client = service.create_sampling_client(base_model=cfg.judge_model)
        judge_renderer = get_renderer(cfg.renderer_name, get_tokenizer(cfg.judge_model))
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_client,
            renderer=judge_renderer,
            max_tokens=cfg.judge_max_tokens,
            actor="judge",
            model_name=cfg.judge_model,
        )

        task = debate_eval(
            adapter=cfg.adapter,
            sampling_client=trained_completer,
            opponent_client=opponent_completer,
            judge_client=judge_completer,
            protocol_kind=cfg.protocol_kind,
            num_rounds=cfg.num_rounds,
            prompts_ref=cfg.prompts_ref,
        )

        log_dir = cfg.log_dir or os.path.expanduser("~/inspect-logs")
        should_log = (self._call_count % cfg.log_evals_every) == 0

        results = await eval_async(
            tasks=[task],
            model=None,
            limit=cfg.limit,
            log_dir=log_dir if should_log else None,
            fail_on_error=False,
            retry_on_error=0,
            log_level="WARNING",
            log_realtime=False,
        )

        # Extract metrics from results.
        metrics: dict[str, float] = {}
        for task_result in results:
            if task_result.results is None or task_result.results.scores is None:
                continue
            for score_result in task_result.results.scores:
                if score_result.name is None:
                    continue
                for _metric_name, metric in score_result.metrics.items():
                    metrics[score_result.name] = metric.value

        logger.info(f"Debate eval metrics (call #{self._call_count}): {metrics}")
        return metrics
