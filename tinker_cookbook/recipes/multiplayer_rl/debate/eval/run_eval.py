"""Offline CLI for debate evaluation on saved checkpoints.

Usage:
    python -m tinker_cookbook.recipes.multiplayer_rl.debate.eval.run_eval \
        model_path=tinker://checkpoint \
        dataset=gpqa \
        limit=50
"""

from __future__ import annotations

import asyncio
import logging

import chz
import tinker

from tinker_cookbook import model_info

from ..progress import run_with_heartbeat
from ..scoring.providers import DebateScorerBuilder
from .dataset_adapter import GPQAAdapter, GPQAOpenEndedAdapter
from .evaluator import DebateInspectEvaluatorBuilder

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    model_path: str | None = None
    model_name: str | None = None
    opponent_model: str | None = None
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    dataset: str = "gpqa"
    dataset_subset: str | None = None
    dataset_split: str = "train"
    record_ids: list[str] | None = chz.field(default_factory=list)
    prompts_ref: str = "scientific_mcq"
    num_rounds: int = 2
    limit: int | None = None
    log_dir: str | None = None
    artifacts_dir: str | None = None
    base_url: str | None = None
    scorer_builder: DebateScorerBuilder | None = None
    scorer_parallelism: int | None = None
    open_reasoning: bool = False
    randomize_position: bool = True
    reasoning_effort: str | None = None
    debater_reasoning_effort: str | None = None
    opponent_reasoning_effort: str | None = None
    judge_reasoning_effort: str | None = None
    opponent_max_tokens: int = 8192
    judge_max_tokens: int = 4096
    heartbeat_seconds: int = 30


async def main(config: Config) -> None:
    logging.basicConfig(level=logging.INFO)

    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Resolve model name from checkpoint if needed.
    model_name = config.model_name
    if model_name is None and config.model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(config.model_path)
        model_name = training_run.base_model
    if model_name is None:
        raise ValueError("model_path or model_name must be provided")

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        model_name,
        reasoning_effort=config.debater_reasoning_effort or config.reasoning_effort,
    )

    # Build adapter.
    if config.dataset == "gpqa":
        adapter = GPQAAdapter(
            subset=config.dataset_subset or "gpqa_diamond",
            limit=config.limit,
        )
    elif config.dataset == "gpqa_open_ended":
        adapter = GPQAOpenEndedAdapter(
            subset=config.dataset_subset or "extended",
            split=config.dataset_split,
            limit=config.limit,
            record_ids=config.record_ids,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Build evaluator (always writes .eval for offline eval).
    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        prompts_ref=config.prompts_ref,
        num_rounds=config.num_rounds,
        opponent_model=config.opponent_model,
        judge_model=config.judge_model,
        opponent_max_tokens=config.opponent_max_tokens,
        judge_max_tokens=config.judge_max_tokens,
        renderer_name=renderer_name,
        reasoning_effort=config.reasoning_effort,
        debater_reasoning_effort=config.debater_reasoning_effort,
        opponent_reasoning_effort=config.opponent_reasoning_effort,
        judge_reasoning_effort=config.judge_reasoning_effort,
        model_name=model_name,
        log_dir=config.artifacts_dir or config.log_dir,
        log_evals_every=1,
        limit=config.limit,
        base_url=config.base_url,
        scorer_builder=config.scorer_builder,
        scorer_parallelism=config.scorer_parallelism,
        open_reasoning=config.open_reasoning,
        randomize_position=config.randomize_position,
    )
    evaluator = builder()

    # Create sampling client for trained model.
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path,
        base_model=model_name,
    )

    metrics = await run_with_heartbeat(
        evaluator(sampling_client),
        label=(
            f"debate eval dataset={config.dataset} "
            f"subset={config.dataset_subset or 'default'} "
            f"limit={config.limit or 'all'}"
        ),
        interval_s=config.heartbeat_seconds,
    )

    logger.info("Debate evaluation completed!")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
