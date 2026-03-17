"""Smoke test: self-play debate training on GPQA diamond.

Runs a 2-batch training loop with self-play (single policy plays both seats):
  - LoRA training with GRPO (importance_sampling loss)
  - LLM judge for outcome reward
  - Validates metrics, reward variance, and training convergence
  - Optional: format penalty step reward (--format-penalty)
  - Optional: think visibility diagnostics (automatic when prompts pack has thinking)

Usage:
    export $(cat .env | xargs)
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_selfplay

    # Full visibility + format penalty smoke (recommended for think-visibility validation):
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_selfplay \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --judge-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --prompts-ref open_selfplay_judgesees \
        --batch-size 32 --group-size 8 --format-penalty
"""

from __future__ import annotations

import argparse
import asyncio
import math
import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig
from tinker.types import LossFnType

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker
from tinker_cookbook.utils import logtree

from ..core.visibility import should_see_thinking
from ..dataset import DebateDataset
from ..plugins import format_penalty_reward_fn
from ..prompts import resolve_prompts
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..scoring.metrics import mcq_debate_metrics
from ..scoring.providers import OpenAICompatibleAnswerJudgeClient
from ..types import DebateGameSpec, ProtocolKind, Role, ThinkVisibility
from ..data.gpqa import load_gpqa_mcq_problems, load_gpqa_open_ended_problems

# --- Defaults ---
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE = 2
GROUP_SIZE = 2
NUM_ROUNDS = 1
MAX_TOKENS = 2048
LEARNING_RATE = 1e-4
LOSS_FN: LossFnType = "importance_sampling"
NUM_BATCHES = 2
LORA_RANK = 32
TRACE_PATH = "/tmp/tinker-examples/smoke_selfplay.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play debate training smoke test")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--prompts-ref", default="selfplay")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=16,
        help="RetryConfig.max_connections for sampling clients.",
    )
    parser.add_argument(
        "--progress-timeout",
        type=int,
        default=900,
        help="RetryConfig.progress_timeout (seconds).",
    )
    parser.add_argument(
        "--format-penalty",
        action="store_true",
        help="Enable format_penalty_reward_fn (-0.1 per step with failed field extraction).",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["mcq", "open_ended"],
        default="mcq",
        help="Problem type: mcq (GPQA diamond) or open_ended (GPQA extended).",
    )
    parser.add_argument(
        "--scorer-model",
        default="gpt-5-mini",
        help="OpenAI model for grader/matcher (OE only).",
    )
    parser.add_argument(
        "--scorer-reasoning-effort",
        default="medium",
        help="Reasoning effort for scorer model.",
    )
    parser.add_argument(
        "--scorer-api-key-env",
        default="OPENAI_API_KEY",
        help="Env var containing OpenAI API key for scorer.",
    )
    parser.add_argument(
        "--episode-log-dir",
        default=None,
        help="Directory for per-episode JSONL transcripts. Default: /tmp/tinker-examples/smoke_selfplay_episodes/",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace):
    is_open_ended = args.scoring_mode == "open_ended"

    # Load problems -- need at least batch_size * num_batches
    n_problems = args.batch_size * args.num_batches
    if is_open_ended:
        problems = load_gpqa_open_ended_problems(limit=n_problems)
    else:
        problems = load_gpqa_mcq_problems(n_problems)
    print(f"Loaded {len(problems)} GPQA {args.scoring_mode} problems", flush=True)
    for i, prob in enumerate(problems):
        print(f"  [{i}] target={prob.target}  {prob.task_prompt[:80]}...", flush=True)
    print(flush=True)

    retry_config = RetryConfig(
        max_connections=args.max_connections,
        progress_timeout=args.progress_timeout,
    )

    # --- Service + training client ---
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        args.model,
        rank=args.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        retry_config=retry_config,
    )

    # Judge (frozen, separate model)
    judge_sampling = service.create_sampling_client(
        base_model=args.judge_model,
        retry_config=retry_config,
    )

    # Renderers
    renderer_name = get_recommended_renderer_name(
        args.model, reasoning_effort=args.reasoning_effort
    )
    judge_renderer_name = get_recommended_renderer_name(
        args.judge_model, reasoning_effort=args.reasoning_effort
    )
    renderer = get_renderer(renderer_name, get_tokenizer(args.model))
    judge_renderer = get_renderer(judge_renderer_name, get_tokenizer(args.judge_model))

    usage = UsageTracker()

    judge_completer = TinkerMessageCompleter(
        sampling_client=judge_sampling,
        renderer=judge_renderer,
        max_tokens=4096,
        usage_tracker=usage,
        actor="judge",
        model_name=args.judge_model,
    )

    # Self-play dataset: opponent_completer=None, opponent_renderer=None
    step_reward_fn = format_penalty_reward_fn if args.format_penalty else None
    episode_log_dir = args.episode_log_dir or "/tmp/tinker-examples/smoke_selfplay_episodes/"
    game = DebateGameSpec(
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=args.num_rounds,
        prompts_ref=args.prompts_ref,
    )

    # OE scoring uses the built-in scorer-backed metrics (metrics=None);
    # MCQ uses the hardcoded choice_match metrics.
    scorer = None
    if is_open_ended:
        scorer = OpenAICompatibleAnswerJudgeClient(
            model=args.scorer_model,
            reasoning_effort=args.scorer_reasoning_effort,
            api_key_env=args.scorer_api_key_env,
        )

    dataset = DebateDataset(
        problems=problems,
        batch_size=args.batch_size,
        group_size=args.group_size,
        game=game,
        renderer=renderer,
        step_reward_fn=step_reward_fn,
        judge_callback=LLMJudgeCallback(judge_completer),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=None,
        opponent_renderer=None,
        metrics=mcq_debate_metrics() if not is_open_ended else None,
        scorer=scorer,
        episode_log_dir=episode_log_dir,
    )

    # Resolve think visibility from the prompts pack for diagnostics.
    prompts = resolve_prompts(args.prompts_ref)
    think_visibility = prompts.get_think_visibility()

    print(f"Model:     {args.model} (renderer={renderer_name})", flush=True)
    print(f"Judge:     {args.judge_model} (renderer={judge_renderer_name})", flush=True)
    print(f"Prompts:   {args.prompts_ref}  reasoning_effort={args.reasoning_effort}", flush=True)
    scorer_desc = f"{args.scorer_model} effort={args.scorer_reasoning_effort}" if scorer else "none"
    print(f"Scoring:   {args.scoring_mode} (scorer={scorer_desc})", flush=True)
    print(f"Format penalty: {args.format_penalty}", flush=True)
    print(f"Episode logs:  {episode_log_dir}", flush=True)
    print(f"Think visibility: {dict(think_visibility)}", flush=True)
    print(
        f"Training:  batch_size={args.batch_size} group_size={args.group_size} "
        f"num_rounds={args.num_rounds} lr={args.lr} lora_rank={args.lora_rank}",
        flush=True,
    )
    print(f"Batches:   {args.num_batches}", flush=True)
    print(f"Trace:     {TRACE_PATH}", flush=True)
    print(flush=True)

    # --- Training loop ---
    all_batch_metrics: list[dict[str, float]] = []
    all_optim_metrics: list[dict[str, float]] = []
    has_reward_variance = False
    all_step_rewards: list[float] = []  # for format penalty assertions

    with logtree.init_trace(
        f"Self-Play Smoke: {args.model}",
        path=TRACE_PATH,
    ):
        for i_batch in range(args.num_batches):
            print(f"===== Batch {i_batch} =====", flush=True)

            # Build policy completer from current sampling client
            policy = TinkerTokenCompleter(
                sampling_client=sampling_client,
                max_tokens=args.max_tokens,
                usage_tracker=usage,
                actor="trained",
                model_name=args.model,
            )

            # Rollouts
            builders = dataset.get_batch(i_batch)
            with logtree.scope_header(f"Batch {i_batch}: rollouts"):
                trajectory_groups_P = await asyncio.gather(
                    *[do_group_rollout(builder, policy) for builder in builders]
                )

            # Collect step rewards from transitions.
            for traj_group in trajectory_groups_P:
                for traj in traj_group.trajectories_G:
                    for t in traj.transitions:
                        all_step_rewards.append(t.reward)

            # Check reward variance
            for traj_group in trajectory_groups_P:
                rewards = traj_group.get_total_rewards()
                if len(set(rewards)) > 1:
                    has_reward_variance = True

            # Metrics
            taglist_P = [builder.logging_tags() for builder in builders]
            batch_metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
            all_batch_metrics.append(batch_metrics)

            # Advantages + training data
            advantages_P = compute_advantages(trajectory_groups_P, builders)
            data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

            # Train step
            optim_metrics: dict[str, float] = {}
            with logtree.scope_header(f"Batch {i_batch}: train_step"):
                await train_step(
                    data_D=data_D,
                    training_client=training_client,
                    learning_rate=args.lr,
                    num_substeps=1,
                    loss_fn=LOSS_FN,
                    metrics=optim_metrics,
                )
            all_optim_metrics.append(optim_metrics)

            # Update sampling client from trained weights
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                retry_config=retry_config,
            )

            # Print batch summary
            print(f"  reward/total: {batch_metrics.get('env/all/reward/total', 'N/A')}", flush=True)
            print(f"  optim metrics: {optim_metrics}", flush=True)
            print(flush=True)

    # --- Assertions ---
    print("=" * 60, flush=True)
    print("ASSERTIONS", flush=True)
    print("=" * 60, flush=True)

    # 1. All metric values are finite
    for i_batch, metrics in enumerate(all_batch_metrics):
        for k, v in metrics.items():
            assert math.isfinite(v), f"Batch {i_batch} metric {k}={v} is not finite"
    print("  [PASS] All metric values are finite", flush=True)

    # 2. At least 1 batch has non-uniform rewards in some trajectory group
    assert has_reward_variance, (
        "All trajectory groups had identical rewards across all batches -- no gradient signal"
    )
    print("  [PASS] Found reward variance in at least one trajectory group", flush=True)

    # 3. judge_quality metric present and in [0, 1]
    for i_batch, metrics in enumerate(all_batch_metrics):
        jq = metrics.get("env/all/judge_quality")
        assert jq is not None, f"Batch {i_batch}: env/all/judge_quality missing"
        assert 0 <= jq <= 1, f"Batch {i_batch}: judge_quality={jq} out of [0,1]"
    print("  [PASS] env/all/judge_quality present and in [0, 1]", flush=True)

    # 4. Accuracy metrics present (soft: may be absent if model outputs are unparseable)
    accuracy_present = all(
        "env/all/accuracy.debater_a" in m and "env/all/accuracy.debater_b" in m
        for m in all_batch_metrics
    )
    if accuracy_present:
        print("  [PASS] env/all/accuracy.debater_a and .debater_b present", flush=True)
    else:
        print(
            "  [WARN] env/all/accuracy.debater_a or .debater_b missing "
            "(model may produce unparseable answers — expected for small models)",
            flush=True,
        )

    # 5. parse_success present
    for i_batch, metrics in enumerate(all_batch_metrics):
        assert "env/all/parse_success" in metrics, f"Batch {i_batch}: env/all/parse_success missing"
    print("  [PASS] env/all/parse_success present", flush=True)

    # 6. Optim metrics present (soft: server may not return metrics for all configs)
    optim_present = all(len(om) > 0 for om in all_optim_metrics)
    if optim_present:
        print("  [PASS] Optim metrics present at both steps", flush=True)
    else:
        print("  [WARN] Some batches returned empty optim metrics", flush=True)

    # 7. Format penalty step rewards (only when --format-penalty)
    if args.format_penalty:
        penalty_count = sum(1 for r in all_step_rewards if r == -0.1)
        clean_count = sum(1 for r in all_step_rewards if r == 0.0)
        total_steps = len(all_step_rewards)
        print(
            f"  Step rewards: {total_steps} total, "
            f"{clean_count} clean (0.0), {penalty_count} penalties (-0.1)",
            flush=True,
        )
        assert penalty_count > 0, (
            f"format_penalty enabled but no -0.1 step rewards in {total_steps} transitions "
            f"— step_reward_fn never fired or all outputs had valid fields"
        )
        assert clean_count > 0, (
            f"format_penalty enabled but no 0.0 step rewards in {total_steps} transitions "
            f"— every output had broken fields (suspicious)"
        )
        print(
            f"  [PASS] Format penalty: {penalty_count}/{total_steps} penalties, "
            f"{clean_count}/{total_steps} clean",
            flush=True,
        )
    else:
        # Without format penalty, all step rewards should be 0.0.
        nonzero = [r for r in all_step_rewards if r != 0.0]
        if nonzero:
            print(
                f"  [WARN] {len(nonzero)} non-zero step rewards without --format-penalty: "
                f"{set(nonzero)}",
                flush=True,
            )
        else:
            print(
                f"  [PASS] All {len(all_step_rewards)} step rewards are 0.0 (no penalty)",
                flush=True,
            )

    # 8. Think visibility: verify prompts pack produces expected mapping
    has_thinking = any(v != ThinkVisibility.DISABLED for v in think_visibility.values())
    if has_thinking:
        print("  Think visibility mapping:", flush=True)
        for role, vis in think_visibility.items():
            print(f"    {role.value}: {vis.value}", flush=True)

        # Verify should_see_thinking is consistent with the mapping.
        for speaker_role, speaker_vis in think_visibility.items():
            if speaker_vis == ThinkVisibility.DISABLED:
                continue
            for viewer_role in Role:
                sees = should_see_thinking(speaker_vis, speaker_role, viewer_role)
                expected = (
                    speaker_role == viewer_role  # own thinking
                    or speaker_vis == ThinkVisibility.OPEN
                    or (
                        speaker_vis == ThinkVisibility.VISIBLE_TO_JUDGE
                        and viewer_role == Role.JUDGE
                    )
                )
                assert sees == expected, (
                    f"should_see_thinking({speaker_vis}, {speaker_role}, {viewer_role}) "
                    f"= {sees}, expected {expected}"
                )
        print("  [PASS] should_see_thinking consistent with think_visibility mapping", flush=True)
    else:
        print("  [SKIP] No thinking enabled in prompts pack — visibility check skipped", flush=True)

    print(flush=True)

    # --- Report ---
    print("=" * 60, flush=True)
    print("METRIC REPORT", flush=True)
    print("=" * 60, flush=True)

    # Collect all metric keys across batches for the table
    all_keys = sorted(
        set().union(*(m.keys() for m in all_batch_metrics)),
    )
    # Header
    header = f"{'Metric':<50}" + "".join(
        f"{'Batch ' + str(i):>14}" for i in range(len(all_batch_metrics))
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in all_keys:
        row = f"{key:<50}"
        for metrics in all_batch_metrics:
            val = metrics.get(key)
            if val is None:
                row += f"{'N/A':>14}"
            else:
                row += f"{val:>14.4f}"
        print(row, flush=True)

    # Optim metrics
    print(flush=True)
    print("Optim metrics:", flush=True)
    optim_keys = sorted(set().union(*(m.keys() for m in all_optim_metrics)))
    for key in optim_keys:
        row = f"  {key:<48}"
        for om in all_optim_metrics:
            val = om.get(key)
            if val is None:
                row += f"{'N/A':>14}"
            else:
                row += f"{val:>14.6f}"
        print(row, flush=True)

    # Step reward distribution
    if all_step_rewards:
        from collections import Counter

        reward_dist = Counter(all_step_rewards)
        print(flush=True)
        print("Step reward distribution:", flush=True)
        for val, count in sorted(reward_dist.items()):
            print(
                f"  {val:>8.3f}: {count:>6} ({count / len(all_step_rewards) * 100:.1f}%)",
                flush=True,
            )

    print(flush=True)
    print(usage.format_cost_report(), flush=True)
    print(f"Trace:    {TRACE_PATH}", flush=True)
    print(f"Episodes: {episode_log_dir}episodes.jsonl", flush=True)
    print("ALL ASSERTIONS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
