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

from .dataset_adapter import GPQAAdapter
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
    prompts_ref: str = "scientific_mcq"
    num_rounds: int = 2
    limit: int | None = None
    log_dir: str | None = None
    base_url: str | None = None


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

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(model_name)

    # Build adapter.
    if config.dataset == "gpqa":
        adapter = GPQAAdapter(limit=config.limit)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Build evaluator (always writes .eval for offline eval).
    builder = DebateInspectEvaluatorBuilder(
        adapter=adapter,
        prompts_ref=config.prompts_ref,
        num_rounds=config.num_rounds,
        opponent_model=config.opponent_model,
        judge_model=config.judge_model,
        renderer_name=renderer_name,
        model_name=model_name,
        log_dir=config.log_dir,
        log_evals_every=1,
        limit=config.limit,
        base_url=config.base_url,
    )
    evaluator = builder()

    # Create sampling client for trained model.
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path,
        base_model=model_name,
    )

    metrics = await evaluator(sampling_client)

    logger.info("Debate evaluation completed!")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
