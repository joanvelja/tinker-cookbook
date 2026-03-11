import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.recipes.gpqa_rl.env import GpqaOpenEndedBuilder
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import AdvantageScheme

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for GPQA open-ended RL training."""

    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    batch_size: int = 16
    group_size: int = 4
    n_batches: int | None = None
    seed: int = 42
    learning_rate: float | None = None
    max_tokens: int = 8192
    temperature: float = 1.0
    advantage_scheme: AdvantageScheme = "maxrl"
    eval_every: int = 5
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    load_checkpoint_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    learning_rate = cli_config.learning_rate or get_lr(cli_config.model_name)
    model_slug = cli_config.model_name.replace("/", "-")
    run_name = f"gpqa_oe-{model_slug}-{cli_config.lora_rank}rank-{learning_rate}lr-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    log_path = cli_config.log_path or f"/tmp/tinker-examples/gpqa_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = GpqaOpenEndedBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        seed=cli_config.seed,
        n_batches=cli_config.n_batches,
    )

    config = Config(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        advantage_scheme=cli_config.advantage_scheme,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
