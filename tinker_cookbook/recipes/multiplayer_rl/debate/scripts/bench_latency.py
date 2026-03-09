"""Latency benchmark for debate sampling topology.

Phase 1: Characterization sweeps — isolated sample_async calls with synthetic prompts.
Phase 2: Topology experiment — full debates under different scheduling/client topologies.

Outputs JSONL with three row types: sweep_call, call, run_summary.

Usage:
    uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.bench_latency \
        --phase all --blocks 3 --model Qwen/Qwen3-4B-Instruct-2507 --output bench.jsonl

    # Quick pilot run
    uv run --env-file .env python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.bench_latency \
        --pilot --phase 2 --arms A0 --output pilot.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

from ..env import DebateEnv
from ..dataset import DebateDataset
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..types import DebateGameSpec, DebateProblemSpec, ProtocolKind
from ..data.gpqa import assign_seat_answers, load_gpqa_mcq_problems

# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


class JSONLWriter:
    """Append-only JSONL output."""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "a")

    def write(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# Phase 1: Characterization Sweeps
# ---------------------------------------------------------------------------

S1_MAX_TOKENS = [2048, 3072, 4096, 8192, 16384]
S2_CONTEXT_LENGTHS = [500, 2000, 4000, 6500, 9500, 12500]
S3_CONCURRENCY = [1, 2, 4, 8, 16]
SWEEP_REPS = 3


async def _timed_sample(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    max_tokens: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Single timed sample_async call. Returns timing dict."""
    t0 = time.monotonic()
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    wall_s = time.monotonic() - t0
    output_tokens = len(result.sequences[0].tokens)
    return {
        "wall_s": wall_s,
        "input_tokens": prompt.length,
        "output_tokens": output_tokens,
        "max_tokens": max_tokens,
    }


def _make_synthetic_prompt(tokenizer: Any, target_length: int) -> tinker.ModelInput:
    """Create a synthetic prompt of approximately target_length tokens."""
    # Repeat a filler sentence and trim to target length.
    filler = "The quick brown fox jumps over the lazy dog. "
    text = filler * (target_length // 5 + 1)
    tokens = tokenizer.encode(text)[:target_length]
    return tinker.ModelInput.from_ints(tokens)


async def run_sweep_s1(
    model: str,
    writer: JSONLWriter,
    warmup: bool,
    run_id: str,
    retry_config: Any | None = None,
) -> None:
    """S1: max_tokens reservation tax."""
    print("=== Sweep S1: max_tokens reservation ===", flush=True)
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=model, retry_config=retry_config)
    tokenizer = get_tokenizer(model)
    prompt = _make_synthetic_prompt(tokenizer, 500)

    if warmup:
        await _timed_sample(client, prompt, max_tokens=256)

    levels = S1_MAX_TOKENS * SWEEP_REPS
    rng = random.Random(42)
    rng.shuffle(levels)

    for i, max_tok in enumerate(levels):
        print(f"  S1 call {i + 1}/{len(levels)}: max_tokens={max_tok}", flush=True)
        try:
            result = await _timed_sample(client, prompt, max_tokens=max_tok)
            writer.write(
                {
                    "type": "sweep_call",
                    "run_id": run_id,
                    "sweep": "S1_max_tokens",
                    "level": max_tok,
                    **result,
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            writer.write(
                {
                    "type": "sweep_call",
                    "run_id": run_id,
                    "sweep": "S1_max_tokens",
                    "level": max_tok,
                    "error": str(e),
                }
            )


async def run_sweep_s2(
    model: str,
    writer: JSONLWriter,
    warmup: bool,
    run_id: str,
    retry_config: Any | None = None,
) -> None:
    """S2: Context length prefill scaling."""
    print("=== Sweep S2: context length prefill ===", flush=True)
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=model, retry_config=retry_config)
    tokenizer = get_tokenizer(model)

    if warmup:
        prompt = _make_synthetic_prompt(tokenizer, 100)
        await _timed_sample(client, prompt, max_tokens=256)

    levels = S2_CONTEXT_LENGTHS * SWEEP_REPS
    rng = random.Random(42)
    rng.shuffle(levels)

    for i, ctx_len in enumerate(levels):
        print(f"  S2 call {i + 1}/{len(levels)}: context_length={ctx_len}", flush=True)
        prompt = _make_synthetic_prompt(tokenizer, ctx_len)
        try:
            result = await _timed_sample(client, prompt, max_tokens=512)
            writer.write(
                {
                    "type": "sweep_call",
                    "run_id": run_id,
                    "sweep": "S2_context_length",
                    "level": ctx_len,
                    **result,
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            writer.write(
                {
                    "type": "sweep_call",
                    "run_id": run_id,
                    "sweep": "S2_context_length",
                    "level": ctx_len,
                    "error": str(e),
                }
            )


async def run_sweep_s3(
    model: str,
    writer: JSONLWriter,
    warmup: bool,
    run_id: str,
    retry_config: Any | None = None,
) -> None:
    """S3: Concurrency saturation (shared vs separate ServiceClient)."""
    print("=== Sweep S3: concurrency saturation ===", flush=True)
    tokenizer = get_tokenizer(model)
    prompt = _make_synthetic_prompt(tokenizer, 500)

    for topology in ("shared", "separate"):
        for n_concurrent in S3_CONCURRENCY:
            for rep in range(SWEEP_REPS):
                label = f"S3 {topology} N={n_concurrent} rep={rep}"
                print(f"  {label}", flush=True)

                if topology == "shared":
                    service = tinker.ServiceClient()
                    clients = [
                        service.create_sampling_client(base_model=model, retry_config=retry_config)
                        for _ in range(n_concurrent)
                    ]
                else:
                    clients = [
                        tinker.ServiceClient().create_sampling_client(
                            base_model=model, retry_config=retry_config
                        )
                        for _ in range(n_concurrent)
                    ]

                if warmup and rep == 0:
                    await _timed_sample(clients[0], prompt, max_tokens=256)

                t0 = time.monotonic()
                try:
                    results = await asyncio.gather(
                        *[_timed_sample(c, prompt, max_tokens=512) for c in clients]
                    )
                    batch_wall_s = time.monotonic() - t0
                    for call_idx, r in enumerate(results):
                        writer.write(
                            {
                                "type": "sweep_call",
                                "run_id": run_id,
                                "sweep": "S3_concurrency",
                                "level": n_concurrent,
                                "topology": topology,
                                "rep": rep,
                                "call_index": call_idx,
                                "batch_wall_s": batch_wall_s,
                                **r,
                            }
                        )
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    writer.write(
                        {
                            "type": "sweep_call",
                            "run_id": run_id,
                            "sweep": "S3_concurrency",
                            "level": n_concurrent,
                            "topology": topology,
                            "rep": rep,
                            "error": str(e),
                        }
                    )


# ---------------------------------------------------------------------------
# Phase 2: Topology Experiment
# ---------------------------------------------------------------------------

ARMS = ["A0", "A1", "A2", "A3"]
N_PROBLEMS_DEFAULT = 2
GROUP_SIZE_DEFAULT = 2
NUM_ROUNDS = 2
PROMPTS_REF_DEFAULT = "scientific_mcq"


@dataclass
class ArmConfig:
    """Topology configuration for one arm."""

    name: str
    parallel_problems: bool
    split_service: bool


ARM_CONFIGS: dict[str, ArmConfig] = {
    "A0": ArmConfig("A0", parallel_problems=False, split_service=False),
    "A1": ArmConfig("A1", parallel_problems=True, split_service=False),
    "A2": ArmConfig("A2", parallel_problems=False, split_service=True),
    "A3": ArmConfig("A3", parallel_problems=True, split_service=True),
}


@dataclass
class RunResult:
    """Result of a single arm run."""

    arm: str
    block: int
    run_id: str
    wall_s: float
    n_problems: int
    group_size: int
    error: str | None = None
    call_rows: list[dict[str, Any]] = field(default_factory=list)


async def _run_arm(
    arm_config: ArmConfig,
    problems: list[DebateProblemSpec],
    model: str,
    judge_model: str,
    group_size: int,
    prompts_ref: str,
    block_idx: int,
    writer: JSONLWriter,
    warmup: bool,
    run_id: str,
    retry_config: Any | None = None,
) -> RunResult:
    """Execute a single arm run: build clients, run debates, collect timing."""
    print(
        f"  Arm {arm_config.name}: parallel={arm_config.parallel_problems}, "
        f"split_service={arm_config.split_service}",
        flush=True,
    )

    renderer = get_renderer("qwen3_instruct", get_tokenizer(model))
    judge_renderer = get_renderer("qwen3_instruct", get_tokenizer(judge_model))

    # Build clients per topology.
    if arm_config.split_service:
        # Separate ServiceClients for debater vs judge.
        debater_service = tinker.ServiceClient()
        judge_service = tinker.ServiceClient()
    else:
        # Shared ServiceClient.
        debater_service = tinker.ServiceClient()
        judge_service = debater_service

    trained_completer = TinkerTokenCompleter(
        sampling_client=debater_service.create_sampling_client(
            base_model=model, retry_config=retry_config
        ),
        max_tokens=3072,
        actor="trained",
        model_name=model,
    )
    opponent_completer = TinkerMessageCompleter(
        sampling_client=debater_service.create_sampling_client(
            base_model=model, retry_config=retry_config
        ),
        renderer=renderer,
        max_tokens=3072,
        actor="opponent",
        model_name=model,
    )
    judge_completer = TinkerMessageCompleter(
        sampling_client=judge_service.create_sampling_client(
            base_model=judge_model, retry_config=retry_config
        ),
        renderer=judge_renderer,
        max_tokens=1024,
        actor="judge",
        model_name=judge_model,
    )

    # Warmup: one call to each model.
    if warmup:
        tokenizer = get_tokenizer(model)
        warm_prompt = _make_synthetic_prompt(tokenizer, 100)
        try:
            await trained_completer.sampling_client.sample_async(
                prompt=warm_prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=32),
            )
        except Exception:
            pass  # warmup failure is non-fatal

    game = DebateGameSpec(
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=NUM_ROUNDS,
        prompts_ref=prompts_ref,
    )
    dataset = DebateDataset(
        problems=problems,
        batch_size=len(problems),
        group_size=group_size,
        game=game,
        renderer=renderer,
        judge_callback=LLMJudgeCallback(judge_completer),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=opponent_completer,
    )

    call_rows: list[dict[str, Any]] = []
    run_t0 = time.monotonic()

    try:
        builders = dataset.get_batch(0)

        async def _run_problem(prob_idx: int, builder: Any) -> list[dict[str, Any]]:
            """Run one problem's envs and collect per-call timing from trajectories."""
            envs = await builder.make_envs()
            trajectories = await asyncio.gather(
                *[do_single_rollout(trained_completer, env) for env in envs]
            )
            await builder.compute_group_rewards(trajectories, envs)

            rows = []
            for env_idx, (env, traj) in enumerate(zip(envs, trajectories)):
                assert isinstance(env, DebateEnv)
                for step_idx, t in enumerate(traj.transitions):
                    row: dict[str, Any] = {
                        "type": "call",
                        "run_id": run_id,
                        "arm": arm_config.name,
                        "block": block_idx,
                        "problem_index": prob_idx,
                        "env_index": env_idx,
                        "step_index": step_idx,
                    }
                    # Pull timing from logs (set by runtime instrumentation).
                    for key in (
                        "role",
                        "phase",
                        "round",
                        "output_tokens",
                        "time/step_wall_s",
                        "time/lock_held_s",
                        "time/judge_wall_s",
                        "time/judge_sample_wall_s",
                        "time/opponent_wall_s",
                        "opponent_output_tokens",
                    ):
                        if key in t.logs:
                            row[key] = t.logs[key]
                    rows.append(row)
            return rows

        if arm_config.parallel_problems:
            # All problems in parallel.
            results = await asyncio.gather(*[_run_problem(i, b) for i, b in enumerate(builders)])
            for rows in results:
                call_rows.extend(rows)
        else:
            # Problems serial.
            for i, builder in enumerate(builders):
                rows = await _run_problem(i, builder)
                call_rows.extend(rows)

        run_wall_s = time.monotonic() - run_t0

        # Write call rows.
        for row in call_rows:
            writer.write(row)

        # Write run summary.
        writer.write(
            {
                "type": "run_summary",
                "run_id": run_id,
                "arm": arm_config.name,
                "block": block_idx,
                "wall_s": run_wall_s,
                "n_problems": len(problems),
                "group_size": group_size,
                "n_calls": len(call_rows),
                "parallel_problems": arm_config.parallel_problems,
                "split_service": arm_config.split_service,
                "model": model,
                "judge_model": judge_model,
            }
        )

        return RunResult(
            arm=arm_config.name,
            block=block_idx,
            run_id=run_id,
            wall_s=run_wall_s,
            n_problems=len(problems),
            group_size=group_size,
            call_rows=call_rows,
        )

    except Exception as e:
        run_wall_s = time.monotonic() - run_t0
        error_str = f"{type(e).__name__}: {e}"
        print(f"    ERROR in arm {arm_config.name}: {error_str}", flush=True)
        traceback.print_exc()
        writer.write(
            {
                "type": "run_summary",
                "run_id": run_id,
                "arm": arm_config.name,
                "block": block_idx,
                "wall_s": run_wall_s,
                "error": error_str,
                "model": model,
                "judge_model": judge_model,
            }
        )
        return RunResult(
            arm=arm_config.name,
            block=block_idx,
            run_id=run_id,
            wall_s=run_wall_s,
            n_problems=len(problems),
            group_size=group_size,
            error=error_str,
        )


async def run_phase2(
    model: str,
    judge_model: str,
    arms: list[str],
    n_blocks: int,
    n_problems: int,
    group_sizes: list[int],
    prompts_ref: str,
    writer: JSONLWriter,
    warmup: bool,
    seed: int,
    retry_config: Any | None = None,
) -> None:
    """Phase 2: blocked randomized topology experiment.

    When group_sizes has multiple entries, each (arm, group_size) pair becomes
    a separate run within each block — this tests H3 (lock contention scaling).
    """
    print(f"=== Phase 2: Topology Experiment ({n_blocks} blocks) ===", flush=True)
    print(f"  Arms: {arms}", flush=True)
    print(f"  Model: {model}", flush=True)
    print(f"  Judge: {judge_model}", flush=True)
    print(f"  Problems: {n_problems}, group_sizes: {group_sizes}", flush=True)
    print(flush=True)

    problems = assign_seat_answers(load_gpqa_mcq_problems(n_problems, seed=seed), seed=seed)
    print(f"  Loaded {len(problems)} GPQA problems", flush=True)

    rng = random.Random(seed)

    # Build (arm, group_size) treatment pairs.
    treatments: list[tuple[str, int]] = [(arm, gs) for arm in arms for gs in group_sizes]

    for block_idx in range(n_blocks):
        block_treatments = list(treatments)
        rng.shuffle(block_treatments)
        order_str = [f"{a}/g{g}" for a, g in block_treatments]
        print(f"\n--- Block {block_idx} (order: {order_str}) ---", flush=True)

        for arm_name, gs in block_treatments:
            arm_config = ARM_CONFIGS[arm_name]
            run_id = uuid.uuid4().hex[:12]
            result = await _run_arm(
                arm_config=arm_config,
                problems=problems,
                model=model,
                judge_model=judge_model,
                group_size=gs,
                prompts_ref=prompts_ref,
                block_idx=block_idx,
                writer=writer,
                warmup=warmup,
                run_id=run_id,
                retry_config=retry_config,
            )
            status = "OK" if result.error is None else f"ERROR: {result.error}"
            gs_label = f" g={gs}" if len(group_sizes) > 1 else ""
            print(
                f"    {arm_name}{gs_label}: {result.wall_s:.1f}s, "
                f"{len(result.call_rows)} calls [{status}]",
                flush=True,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debate sampling latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase to run",
    )
    p.add_argument("--blocks", type=int, default=3, help="Number of Phase 2 blocks")
    p.add_argument(
        "--arms",
        type=str,
        default="A0,A1,A2,A3",
        help="Comma-separated arm list for Phase 2",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Debater model",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model (defaults to --model)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="bench_latency.jsonl",
        help="JSONL output path",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Quick mode: 1 block, 1 problem, group_size=1",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup calls",
    )
    p.add_argument("--seed", type=int, default=42, help="PRNG seed")
    p.add_argument(
        "--prompts-ref",
        type=str,
        default=PROMPTS_REF_DEFAULT,
        help="Prompts reference for debate",
    )
    p.add_argument(
        "--n-problems",
        type=int,
        default=N_PROBLEMS_DEFAULT,
        help="Number of GPQA problems",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=GROUP_SIZE_DEFAULT,
        help="Envs per problem (single value, used unless --group-sizes is set)",
    )
    p.add_argument(
        "--group-sizes",
        type=str,
        default=None,
        help="Comma-separated group sizes to sweep in Phase 2 (e.g. '1,2,4'). "
        "Overrides --group-size. Each (arm, group_size) pair is a run.",
    )
    p.add_argument(
        "--retry-timeout",
        type=int,
        default=None,
        help="Override RetryConfig.progress_timeout (seconds). Default: SDK default (7200s).",
    )
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    judge_model = args.judge_model or args.model
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARM_CONFIGS:
            print(f"ERROR: unknown arm '{a}'. Valid: {list(ARM_CONFIGS)}", flush=True)
            sys.exit(1)

    warmup = not args.no_warmup

    # Build retry config if overridden.
    retry_config = None
    if args.retry_timeout is not None:
        from tinker.lib.public_interfaces.service_client import RetryConfig

        retry_config = RetryConfig(progress_timeout=args.retry_timeout)
        print(f"RetryConfig: progress_timeout={args.retry_timeout}s", flush=True)

    n_problems = args.n_problems
    n_blocks = args.blocks

    # Resolve group sizes: --group-sizes overrides --group-size.
    if args.group_sizes is not None:
        group_sizes = [int(g.strip()) for g in args.group_sizes.split(",")]
    else:
        group_sizes = [args.group_size]

    if args.pilot:
        n_problems = max(n_problems, 2)  # >=2 so parallel vs serial is a real treatment
        group_sizes = [1]
        n_blocks = 1
        print(
            f"PILOT MODE: {n_blocks} block, {n_problems} problems, group_sizes={group_sizes}",
            flush=True,
        )

    output_path = Path(args.output)
    writer = JSONLWriter(output_path)

    # Write session header.
    session_id = uuid.uuid4().hex[:12]
    writer.write(
        {
            "type": "session",
            "session_id": session_id,
            "model": args.model,
            "judge_model": judge_model,
            "arms": arms,
            "n_blocks": n_blocks,
            "n_problems": n_problems,
            "group_sizes": group_sizes,
            "seed": args.seed,
            "prompts_ref": args.prompts_ref,
            "pilot": args.pilot,
            "warmup": warmup,
            "retry_timeout": args.retry_timeout,
            "timestamp": time.time(),
        }
    )

    print(f"Output: {output_path}", flush=True)
    print(f"Session: {session_id}", flush=True)
    print(flush=True)

    try:
        if args.phase in ("1", "all"):
            run_id = uuid.uuid4().hex[:12]
            await run_sweep_s1(args.model, writer, warmup, run_id, retry_config)
            run_id = uuid.uuid4().hex[:12]
            await run_sweep_s2(args.model, writer, warmup, run_id, retry_config)
            run_id = uuid.uuid4().hex[:12]
            await run_sweep_s3(args.model, writer, warmup, run_id, retry_config)

        if args.phase in ("2", "all"):
            await run_phase2(
                model=args.model,
                judge_model=judge_model,
                arms=arms,
                n_blocks=n_blocks,
                n_problems=n_problems,
                group_sizes=group_sizes,
                prompts_ref=args.prompts_ref,
                writer=writer,
                warmup=warmup,
                seed=args.seed,
                retry_config=retry_config,
            )
    finally:
        writer.close()

    print(f"\nDone. Results: {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
