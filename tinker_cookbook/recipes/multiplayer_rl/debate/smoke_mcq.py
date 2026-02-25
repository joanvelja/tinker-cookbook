"""Smoke test: scientific MCQ debate with GPQA diamond.

Runs 2 problems × group_size=2 with frozen opponent + LLM judge.
Verifies the full scoring pipeline: field extraction, trajectory queries,
metrics computation, reward flattening.

Writes HTML trace to /tmp/tinker-examples/smoke_mcq.html

Usage:
    export $(cat .env | xargs)
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.smoke_mcq
"""

from __future__ import annotations

import asyncio
import random

import datasets
import tinker

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker
from tinker_cookbook.utils import logtree

from .env import DebateDataset, DebateEnv, DebateProblem
from .judge import LLMJudgeCallback, zero_sum_outcome_reward
from .metrics import mcq_debate_metrics
from .trace_fmt import DebateTraceCSSInjector, render_rollout_html
from .types import ProtocolKind

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER = "qwen3_instruct"
PROMPTS_REF = "scientific_mcq"
TRACE_PATH = "/tmp/tinker-examples/smoke_mcq.html"
N_PROBLEMS = 2


def _load_gpqa_problems(n: int, seed: int = 42) -> list[DebateProblem]:
    """Load n GPQA diamond problems as (task_prompt, answer_a, answer_b, target) tuples."""
    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    problems: list[DebateProblem] = []
    for idx in indices:
        row = ds[idx]
        correct = row["Correct Answer"]
        wrong = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]

        # Shuffle into ABCD, track correct label.
        options = [correct] + wrong
        rng.shuffle(options)
        target_label = chr(ord("A") + options.index(correct))

        # Format as MCQ prompt.
        question = row["Question"]
        option_lines = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
        task_prompt = f"{question}\n\n{option_lines}"

        # Debater A gets the correct answer, debater B gets a random wrong one.
        wrong_label = rng.choice([chr(ord("A") + i) for i in range(4) if chr(ord("A") + i) != target_label])
        problems.append((task_prompt, target_label, wrong_label, target_label))

    return problems


async def run():
    problems = _load_gpqa_problems(N_PROBLEMS)
    print(f"Loaded {len(problems)} GPQA problems")
    for i, (prompt, ans_a, ans_b, target) in enumerate(problems):
        print(f"  [{i}] A={ans_a} B={ans_b} target={target}  {prompt[:80]}...")
    print()

    service = tinker.ServiceClient()
    renderer = get_renderer(RENDERER, get_tokenizer(MODEL))
    usage = UsageTracker()

    opponent_completer = TinkerMessageCompleter(
        sampling_client=service.create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=2048,
        usage_tracker=usage,
        actor="opponent",
        model_name=MODEL,
    )
    judge_completer = TinkerMessageCompleter(
        sampling_client=service.create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=1024,
        usage_tracker=usage,
        actor="judge",
        model_name=MODEL,
    )
    trained_completer = TinkerTokenCompleter(
        sampling_client=service.create_sampling_client(base_model=MODEL),
        max_tokens=2048,
        usage_tracker=usage,
        actor="trained",
        model_name=MODEL,
    )

    dataset = DebateDataset(
        problems=problems,
        batch_size=len(problems),
        renderer=renderer,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=2,
        judge_callback=LLMJudgeCallback(judge_completer),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=opponent_completer,
        group_size=2,
        prompts_ref=PROMPTS_REF,
        metrics=mcq_debate_metrics(),
    )

    print(f"Model: {MODEL}")
    print(f"Prompts: {PROMPTS_REF}")
    print(f"Trace: {TRACE_PATH}")
    print()

    with logtree.init_trace(f"MCQ Debate Smoke: {PROMPTS_REF} / {MODEL}", path=TRACE_PATH):
        logtree.log_formatter(DebateTraceCSSInjector())
        logtree.log_text(
            f"Model: {MODEL} | Prompts: {PROMPTS_REF} | "
            f"Protocol: SEQUENTIAL | Rounds: 2 | Group size: 2 | "
            f"Problems: {N_PROBLEMS}"
        )

        builders = dataset.get_batch(0)
        for i, builder in enumerate(builders):
            with logtree.scope_header(
                f"Problem {i}: {builder.task_prompt[:80]}...",
                class_="lt-section db-group",
            ):
                envs = await builder.make_envs()

                with logtree.scope_disable():
                    trajectories = await asyncio.gather(
                        *[do_single_rollout(trained_completer, env) for env in envs]
                    )

                rewards_and_metrics = await builder.compute_group_rewards(trajectories, envs)

                # Log HTML trace.
                for env, traj, (reward, metrics) in zip(envs, trajectories, rewards_and_metrics):
                    assert isinstance(env, DebateEnv)
                    logtree.log_html(render_rollout_html(env, reward))

                # Print metrics.
                for env_idx, (env, (reward, metrics)) in enumerate(zip(envs, rewards_and_metrics)):
                    assert isinstance(env, DebateEnv)
                    print(f"  Problem {i} / env {env_idx} ({env.role.value}): reward={reward:.1f}")
                    for k, v in sorted(metrics.items()):
                        print(f"    {k}: {v}")
                    print()

        with logtree.scope_header("Cost Report", class_="lt-section db-cost"):
            logtree.details(usage.format_cost_report(), summary="Tinker Usage", pre=True)

    print(usage.format_cost_report())
    print(f"Trace: {TRACE_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
