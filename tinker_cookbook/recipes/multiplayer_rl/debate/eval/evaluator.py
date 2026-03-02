"""Evaluator that runs debate evals via Inspect AI."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import chz
import tinker
from inspect_ai import eval_async
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

if TYPE_CHECKING:
    from tinker_cookbook.usage import UsageTracker

from ..types import ProtocolKind
from .dataset_adapter import DatasetAdapter
from .inspect_task import debate_eval

logger = logging.getLogger(__name__)


@chz.chz
class DebateInspectEvaluatorBuilder:
    """Config for debate evaluation via Inspect AI."""

    adapter: DatasetAdapter
    prompts_ref: str = "judge_exploit"
    num_rounds: int = 2
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    open_reasoning: bool = False
    randomize_position: bool = True

    opponent_model: str | None = None  # None = self-play
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    opponent_max_tokens: int = 8192
    judge_max_tokens: int = 4096
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    model_name: str | None = None

    log_dir: str | None = None
    log_evals_every: int = 1
    limit: int | None = None
    base_url: str | None = None
    max_connections: int = 256
    progress_timeout_s: int = 900

    def __call__(self) -> DebateInspectEvaluator:
        return DebateInspectEvaluator(self)


class DebateInspectEvaluator(SamplingClientEvaluator):
    """Runs debate evaluation, optionally writing .eval files via Inspect."""

    def __init__(self, config: DebateInspectEvaluatorBuilder) -> None:
        self._config = config
        self._call_count = 0

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        usage_tracker: UsageTracker | None = None,
    ) -> dict[str, float]:
        cfg = self._config
        self._call_count += 1
        retry_config = RetryConfig(
            max_connections=cfg.max_connections,
            progress_timeout=cfg.progress_timeout_s,
        )

        if cfg.model_name is None:
            raise ValueError("model_name must be set on DebateInspectEvaluatorBuilder")

        # Per-role renderer construction.
        trained_name = cfg.renderer_name or model_info.get_recommended_renderer_name(
            cfg.model_name, reasoning_effort=cfg.reasoning_effort
        )
        trained_renderer = get_renderer(trained_name, get_tokenizer(cfg.model_name))

        # Trained model completer (debater A).
        trained_completer = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=trained_renderer,
            max_tokens=cfg.opponent_max_tokens,
            usage_tracker=usage_tracker,
            actor="trained",
            model_name=cfg.model_name,
        )

        # Separate ServiceClient for non-trained actors.
        service = tinker.ServiceClient(base_url=cfg.base_url)

        # Opponent completer. Self-play shares the trained sampling_client
        # (same fine-tuned weights). Separate model gets its own client.
        if cfg.opponent_model is None:
            opponent_completer = TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=trained_renderer,
                max_tokens=cfg.opponent_max_tokens,
                usage_tracker=usage_tracker,
                actor="opponent",
                model_name=cfg.model_name,
            )
        else:
            opp_name = model_info.get_recommended_renderer_name(
                cfg.opponent_model, reasoning_effort=cfg.reasoning_effort
            )
            opp_renderer = get_renderer(opp_name, get_tokenizer(cfg.opponent_model))
            opp_client = service.create_sampling_client(
                base_model=cfg.opponent_model,
                retry_config=retry_config,
            )
            opponent_completer = TinkerMessageCompleter(
                sampling_client=opp_client,
                renderer=opp_renderer,
                max_tokens=cfg.opponent_max_tokens,
                usage_tracker=usage_tracker,
                actor="opponent",
                model_name=cfg.opponent_model,
            )

        # Judge completer — always gets its own sampling client.
        judge_name = model_info.get_recommended_renderer_name(
            cfg.judge_model, reasoning_effort=cfg.reasoning_effort
        )
        judge_renderer = get_renderer(judge_name, get_tokenizer(cfg.judge_model))
        judge_client = service.create_sampling_client(
            base_model=cfg.judge_model,
            retry_config=retry_config,
        )
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_client,
            renderer=judge_renderer,
            max_tokens=cfg.judge_max_tokens,
            usage_tracker=usage_tracker,
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
            open_reasoning=cfg.open_reasoning,
            randomize_position=cfg.randomize_position,
        )

        log_dir = cfg.log_dir or os.path.expanduser("~/inspect-logs")
        should_log = (self._call_count % cfg.log_evals_every) == 0

        # Inspect AI always writes .eval files — log_dir=None falls through to
        # ./logs/ default. Use a tempdir as a trash sink when we don't want logs.
        if should_log:
            results = await eval_async(
                tasks=[task],
                model=None,
                limit=cfg.limit,
                max_connections=cfg.max_connections,
                log_dir=log_dir,
                fail_on_error=False,
                retry_on_error=0,
                log_level="WARNING",
                log_realtime=False,
            )
        else:
            with tempfile.TemporaryDirectory() as trash_dir:
                results = await eval_async(
                    tasks=[task],
                    model=None,
                    limit=cfg.limit,
                    max_connections=cfg.max_connections,
                    log_dir=trash_dir,
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
                for metric_name, metric in score_result.metrics.items():
                    metrics[metric_name] = metric.value

        logger.info(f"Debate eval metrics (call #{self._call_count}): {metrics}")
        return metrics
