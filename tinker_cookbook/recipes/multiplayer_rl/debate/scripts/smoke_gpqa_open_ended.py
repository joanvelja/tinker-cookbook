"""Smoke test: GPQA open-ended self-play with semantic scorer artifacts.

Runs a tiny self-play debate batch on GPQA open-ended using:
  - gpt-oss-120b debaters
  - gpt-oss-20b judge
  - gpt-5-mini semantic matcher/grader

Artifacts are written under artifacts/debate/gpqa_open_ended/<run_name>/.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker
from tinker_cookbook.utils import logtree

from ..env import DebateEnv
from ..dataset import DebateDataset
from ..data.gpqa import load_gpqa_open_ended_problems, problem_to_sample
from ..progress import run_with_heartbeat
from ..scoring import AnswerJudgeClient, DebateScorerBuilder, RecordingAnswerJudgeClient
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..types import DebateGameSpec, ProtocolKind
from .trace_fmt import DebateTraceCSSInjector, render_rollout_html

DEBATER_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-20b"
SCORER_MODEL = "gpt-5-mini"
DEFAULT_RECORD_IDS = [
    "rec1BjNQici8oD53a",
    "recxm2fjqvae5dDYW",
]


def _default_run_name(prompts_ref: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{prompts_ref}"


def _artifact_dir(root: str | None, prompts_ref: str) -> Path:
    base = Path(root or "artifacts/debate/gpqa_open_ended")
    return base / _default_run_name(prompts_ref)


def _render_scorer_calls_html(problem_calls) -> str:
    parts = ['<div class="db-judge"><div class="db-judge-hdr">Semantic Scorer Calls</div>']
    if not problem_calls:
        parts.append('<div class="db-meta">No scorer calls for this problem (all fast-path).</div>')
        parts.append("</div>")
        return "\n".join(parts)

    for idx, call in enumerate(problem_calls, start=1):
        parts.append('<details class="db-io">')
        parts.append(
            f"<summary>Call {idx}: {html.escape(call.kind or 'unknown')} → "
            f"{html.escape(call.response.strip() or '<empty>')}</summary>"
        )
        parts.append(f'<pre class="db-pre">{html.escape(call.system)}</pre>')
        parts.append(f'<pre class="db-pre">{html.escape(call.user)}</pre>')
        parts.append(f'<pre class="db-pre">{html.escape(call.response)}</pre>')
        parts.append("</details>")
    parts.append("</div>")
    return "\n".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPQA open-ended semantic smoke")
    parser.add_argument("--subset", default="extended")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--record-id",
        dest="record_ids",
        action="append",
        default=[],
        help="Repeatable record id selector. Defaults to two fixed smoke records.",
    )
    parser.add_argument("--prompts-ref", default="open_balanced")
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--debater-model", default=DEBATER_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--scorer-model", default=SCORER_MODEL)
    parser.add_argument("--debater-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--scorer-max-tokens", type=int, default=16_384)
    parser.add_argument("--debater-reasoning-effort", default="high")
    parser.add_argument("--judge-reasoning-effort", default="medium")
    parser.add_argument("--scorer-reasoning-effort", default="high")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--scorer-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--progress-timeout", type=int, default=900)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    artifact_dir = _artifact_dir(args.artifacts_dir, args.prompts_ref)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "trace.html"
    episode_log_dir = artifact_dir / "episodes"

    record_ids = args.record_ids or list(DEFAULT_RECORD_IDS)
    problems = load_gpqa_open_ended_problems(
        subset=args.subset,
        split=args.split,
        record_ids=record_ids,
    )
    samples = [problem_to_sample(p, source="gpqa_open_ended") for p in problems]

    retry_config = RetryConfig(
        max_connections=args.max_connections,
        progress_timeout=args.progress_timeout,
    )
    service = tinker.ServiceClient(base_url=args.base_url)
    usage = UsageTracker()

    sampling_client = service.create_sampling_client(
        base_model=args.debater_model,
        retry_config=retry_config,
    )
    judge_sampling = service.create_sampling_client(
        base_model=args.judge_model,
        retry_config=retry_config,
    )

    debater_renderer_name = get_recommended_renderer_name(
        args.debater_model, reasoning_effort=args.debater_reasoning_effort
    )
    judge_renderer_name = get_recommended_renderer_name(
        args.judge_model, reasoning_effort=args.judge_reasoning_effort
    )
    debater_renderer = get_renderer(debater_renderer_name, get_tokenizer(args.debater_model))
    judge_renderer = get_renderer(judge_renderer_name, get_tokenizer(args.judge_model))

    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=args.debater_max_tokens,
        usage_tracker=usage,
        actor="trained",
        model_name=args.debater_model,
    )
    judge = TinkerMessageCompleter(
        sampling_client=judge_sampling,
        renderer=judge_renderer,
        max_tokens=args.judge_max_tokens,
        usage_tracker=usage,
        actor="judge",
        model_name=args.judge_model,
    )
    scorer = RecordingAnswerJudgeClient(
        DebateScorerBuilder(
            provider="openai_compatible",
            model=args.scorer_model,
            base_url=args.scorer_base_url,
            api_key_env="OPENAI_API_KEY",
            reasoning_effort=args.scorer_reasoning_effort,
            max_tokens=args.scorer_max_tokens,
            max_connections=args.max_connections,
        ).build(usage_tracker=usage)
    )

    game = DebateGameSpec(
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=args.num_rounds,
        prompts_ref=args.prompts_ref,
    )
    dataset = DebateDataset(
        problems=problems,
        batch_size=len(problems),
        group_size=args.group_size,
        game=game,
        renderer=debater_renderer,
        judge_callback=LLMJudgeCallback(judge),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=None,
        opponent_renderer=None,
        randomize_position=False,
        scorer=scorer,
        scorer_parallelism=args.max_connections,
        episode_log_dir=str(episode_log_dir),
    )

    config_path = artifact_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "subset": args.subset,
                "split": args.split,
                "record_ids": record_ids,
                "prompts_ref": args.prompts_ref,
                "debater_model": args.debater_model,
                "judge_model": args.judge_model,
                "scorer_model": args.scorer_model,
                "debater_max_tokens": args.debater_max_tokens,
                "judge_max_tokens": args.judge_max_tokens,
                "scorer_max_tokens": args.scorer_max_tokens,
                "debater_reasoning_effort": args.debater_reasoning_effort,
                "judge_reasoning_effort": args.judge_reasoning_effort,
                "scorer_reasoning_effort": args.scorer_reasoning_effort,
                "heartbeat_seconds": args.heartbeat_seconds,
            },
            indent=2,
        )
    )

    print(f"Artifacts: {artifact_dir}", flush=True)
    print(f"Trace:     {trace_path}", flush=True)
    print(f"Prompts:   {args.prompts_ref}", flush=True)
    print(f"Records:   {record_ids}", flush=True)
    print(f"Heartbeat: every {args.heartbeat_seconds}s", flush=True)
    print(flush=True)

    builders = dataset.get_batch(0)
    summaries: list[dict[str, object]] = []

    async def _run_problem(
        idx: int,
        sample,
        builder,
        inner_scorer: AnswerJudgeClient,
        heartbeat_s: int,
    ):
        recorder = RecordingAnswerJudgeClient(inner_scorer)
        builder.scorer = recorder
        n = len(samples)
        record_id = str(sample.metadata["record_id"])
        envs = await run_with_heartbeat(
            builder.make_envs(),
            label=f"problem {idx + 1}/{n} {record_id}: make_envs",
            interval_s=heartbeat_s,
        )
        trajectories = await run_with_heartbeat(
            asyncio.gather(*[do_single_rollout(policy, env) for env in envs]),
            label=f"problem {idx + 1}/{n} {record_id}: rollouts",
            interval_s=heartbeat_s,
        )
        rewards_and_metrics = await run_with_heartbeat(
            builder.compute_group_rewards(trajectories, envs),
            label=f"problem {idx + 1}/{n} {record_id}: rewards+metrics",
            interval_s=heartbeat_s,
        )
        return idx, sample, builder, envs, trajectories, rewards_and_metrics, recorder.calls

    try:
        with logtree.init_trace(f"GPQA Open-Ended Smoke: {args.prompts_ref}", path=str(trace_path)):
            logtree.log_formatter(DebateTraceCSSInjector())

            results = await asyncio.gather(*[
                _run_problem(idx, sample, builder, scorer.inner, args.heartbeat_seconds)
                for idx, (sample, builder) in enumerate(zip(samples, builders, strict=True))
            ])

            all_scorer_calls = []
            for idx, sample, builder, envs, trajectories, rewards_and_metrics, problem_calls in results:
                builder.on_group_complete(trajectories, envs, rewards_and_metrics)
                all_scorer_calls.extend(problem_calls)

                with logtree.scope_header(
                    f"Problem {idx}: {sample.metadata['record_id']} / {sample.input[:80]}...",
                    class_="lt-section db-group",
                ):
                    for env, (reward, metrics) in zip(envs, rewards_and_metrics, strict=True):
                        assert isinstance(env, DebateEnv)
                        logtree.log_html(render_rollout_html(env, reward))
                        summaries.append(
                            {
                                "record_id": sample.metadata["record_id"],
                                "role": env.role.value,
                                "reward": reward,
                                "metrics": metrics,
                                "scorer_call_count": len(problem_calls),
                            }
                        )
                    logtree.log_html(_render_scorer_calls_html(problem_calls))

            scorer.calls = all_scorer_calls
    finally:
        scorer.write_jsonl(artifact_dir / "semantic_calls.jsonl")
        (artifact_dir / "summary.json").write_text(json.dumps(summaries, indent=2))


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
