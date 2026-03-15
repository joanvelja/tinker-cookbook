"""Debate environment builders (group + branch)."""

from __future__ import annotations

import json
import os
import random
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    Trajectory,
)
from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import Renderer

from .scoring.metrics import MetricFn, mcq_debate_metrics
from .scoring.trajectory import final_answer
from .plugins import JudgeCallback, OutcomeRewardFn, StepRewardFn
from .prompts import check_ab_symmetry, resolve_prompts
from .core.reducer import fork_state
from .core.runtime import DebateRuntime
from .core.schedule import build_schedule
from .types import (
    DebateGameSpec,
    DebateProblemSpec,
    DebateSnapshot,
    DebateSpec,
    DebateState,
    Role,
    ScoringMode,
)
from .env import DebateEnv

if TYPE_CHECKING:
    from .scoring.providers import AnswerJudgeClient

_EPISODE_LOG_LOCK = threading.Lock()

IDENTITY_REMAP_BASES = [
    "accuracy",
    "stance_change",
    "concession_correctness",
    "debater_accuracy_delta",
    "win_rate",
    "loss_rate",
    "correct_and_wins",
    "correct_and_loses",
    "wrong_and_wins",
    "wrong_and_loses",
    "parse_success",
    "think_block_rate",
]


def _remap_to_identity(m: Metrics, trained_role: Role) -> Metrics:
    """Remap seat-based metrics (debater_a/b) to identity-based (trained/opponent).

    Returns a NEW dict with all original keys plus id/ prefixed identity keys.
    """
    result = dict(m)  # keep originals

    is_a = trained_role == Role.DEBATER_A
    result["id/trained_role_is_a"] = 1.0 if is_a else 0.0

    for base in IDENTITY_REMAP_BASES:
        a_key = f"{base}.debater_a"
        b_key = f"{base}.debater_b"
        trained_key = f"id/{base}.trained"
        opponent_key = f"id/{base}.opponent"

        if is_a:
            if a_key in m:
                result[trained_key] = m[a_key]
            if b_key in m:
                result[opponent_key] = m[b_key]
        else:
            if b_key in m:
                result[trained_key] = m[b_key]
            if a_key in m:
                result[opponent_key] = m[a_key]

    return result


def _compute_metrics_sync(
    state: DebateState,
    metrics: dict[str, MetricFn] | None,
) -> Metrics:
    metric_fns = metrics if metrics is not None else mcq_debate_metrics()
    results = {name: fn(state) for name, fn in metric_fns.items()}
    return {name: r.value for name, r in results.items() if r.value is not None}


async def _compute_builtin_metrics_batch(
    states: Sequence[DebateState],
    scorer: AnswerJudgeClient | None,
    scorer_parallelism: int,
) -> list[Metrics]:
    from .scoring.facts import built_in_metric_values, resolve_debate_facts_for_states

    facts_by_state = await resolve_debate_facts_for_states(
        states,
        scorer=scorer,
        prompts_for_ref=resolve_prompts,
        parallelism=scorer_parallelism,
        strict=False,
    )
    return [
        {
            name: value
            for name, value in built_in_metric_values(state, facts).items()
            if value is not None
        }
        for state, facts in zip(states, facts_by_state, strict=True)
    ]


async def _compute_metrics(
    states: Sequence[DebateState],
    metrics: dict[str, MetricFn] | None,
    scorer: AnswerJudgeClient | None,
    scorer_parallelism: int,
) -> list[Metrics]:
    """Dispatch to builtin batch scorer or sync per-state MetricFns."""
    if metrics is None:
        return await _compute_builtin_metrics_batch(states, scorer, scorer_parallelism)
    return [_compute_metrics_sync(state, metrics) for state in states]


@dataclass
class DebateGroupBuilder(EnvGroupBuilder):
    """Builds a group of DebateEnvs.

    Both modes create group_size independent runtimes stored in _runtimes.

    In normal/self-play mode (opponent_completer=None), each runtime gets
    len(include_roles) envs — all sharing the same coordinator.

    In frozen-opponent mode (opponent_completer set), each runtime gets a
    single trained-agent env. The opponent is driven by the completer via
    _drive_opponent().
    """

    problem: DebateProblemSpec
    game: DebateGameSpec
    renderer: Renderer
    step_reward_fn: StepRewardFn | None = None
    judge_callback: JudgeCallback | None = None
    outcome_reward_fn: OutcomeRewardFn | None = None
    include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B)
    group_size: int = 1
    opponent_completer: MessageCompleter | None = None
    opponent_renderer: Renderer | None = None
    randomize_position: bool = False
    metrics: dict[str, MetricFn] | None = field(default=None, repr=False)
    scorer: AnswerJudgeClient | None = field(default=None, repr=False)
    scorer_parallelism: int = 64
    episode_log_dir: str | None = None
    split: str | None = None
    step: int | None = None

    # Set after make_envs
    _runtimes: list[DebateRuntime] = field(default_factory=list, repr=False)

    def on_group_complete(
        self,
        trajectories_G: list[Trajectory],
        env_group: Sequence[Env],
        rewards_and_metrics_G: list[tuple[float, Metrics]],
    ) -> None:
        if self.episode_log_dir is None:
            return

        is_selfplay = self.opponent_completer is None

        os.makedirs(self.episode_log_dir, exist_ok=True)
        records: list[str] = []

        for env, (reward, metrics) in zip(env_group, rewards_and_metrics_G, strict=True):
            assert isinstance(env, DebateEnv)
            state = env.runtime.state
            this_role = env.role
            other_role = Role.DEBATER_B if this_role == Role.DEBATER_A else Role.DEBATER_A

            # Shared fields across both modes.
            record: dict = {
                "schema_version": 3 if is_selfplay else 2,
                "step": self.step,
                "split": self.split,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "debate_id": state.spec.debate_id,
                "protocol_kind": state.spec.protocol_kind.value,
                "prompts_ref": state.spec.prompts_ref,
                "think_visibility": state.spec.encode_think_visibility(),
                "target": state.spec.problem.target,
                "winner": (
                    state.outcome.winner.value if state.outcome and state.outcome.winner else None
                ),
                "verdict_text": state.outcome.verdict_text if state.outcome else None,
            }

            if is_selfplay:
                # Self-play: seat-based framing. No trained/opponent distinction.
                record.update(
                    {
                        "role": this_role.value,
                        "reward": reward,
                        "answers": {
                            f"public_{Role.DEBATER_A.value}": final_answer(
                                state, role=Role.DEBATER_A
                            ),
                            f"public_{Role.DEBATER_B.value}": final_answer(
                                state, role=Role.DEBATER_B
                            ),
                        },
                        "signals": dict(metrics),
                    }
                )
            else:
                # Frozen-opponent: identity-based framing.
                record.update(
                    {
                        "trained_role": this_role.value,
                        "reward_trained": reward,
                        "answers": {
                            "public_trained": final_answer(state, role=this_role),
                            "public_opponent": final_answer(state, role=other_role),
                        },
                        "signals": {k: v for k, v in metrics.items() if k.startswith("id/")},
                    }
                )

            record["transcript"] = [
                {
                    "role": utt.role.value,
                    "phase": utt.phase.value,
                    "round": utt.round_index,
                    "text": utt.text,
                    "identity": (
                        utt.role.value
                        if is_selfplay
                        else ("trained" if utt.role == env.role else "opponent")
                    ),
                }
                for utt in state.transcript
            ]
            records.append(json.dumps(record) + "\n")

        log_path = os.path.join(self.episode_log_dir, "episodes.jsonl")
        with _EPISODE_LOG_LOCK:
            with open(log_path, "a") as f:
                f.writelines(records)

    async def make_envs(self) -> Sequence[Env]:
        if self.opponent_completer is None and self.randomize_position:
            raise ValueError(
                "randomize_position=True has no effect in self-play mode "
                "(both agents share the same policy). Set randomize_position=False."
            )

        # Eager validation: fail fast on bad prompts_ref.
        prompts = resolve_prompts(self.game.prompts_ref)
        if self.problem.scoring_mode == ScoringMode.OPEN_ENDED:
            if self.scorer is None:
                raise ValueError(
                    "OPEN_ENDED debate scoring requires a scorer client. "
                    "Pass scorer=... on DebateGroupBuilder."
                )
            if prompts.get_binary_judge_template("matcher") is None:
                raise ValueError(
                    f"OPEN_ENDED debate scoring requires _matcher in {prompts.source_ref}."
                )
            if prompts.get_binary_judge_template("grader") is None:
                raise ValueError(
                    f"OPEN_ENDED debate scoring requires _grader in {prompts.source_ref}."
                )
        if self.randomize_position:
            for w in check_ab_symmetry(prompts):
                warnings.warn(f"randomize_position=True but A/B asymmetry: {w}")

        schedule = build_schedule(
            self.game.protocol_kind,
            self.game.num_rounds,
            include_judge_turns=self.game.include_judge_turns,
        )

        if self.opponent_completer is not None:
            # Frozen-opponent mode: validate schedule actors fit in {trained, opponent}.
            all_schedule_roles = {r for slot in schedule for r in slot.actors}
            debater_roles = {Role.DEBATER_A, Role.DEBATER_B}
            extra = all_schedule_roles - debater_roles
            if extra:
                raise ValueError(
                    f"Frozen-opponent mode only drives debater roles, but schedule "
                    f"includes {extra}. Set include_judge_turns=False or use normal mode."
                )
            envs: list[Env] = []
            self._runtimes = []
            for _ in range(self.group_size):
                if self.randomize_position:
                    trained_role = random.choice([Role.DEBATER_A, Role.DEBATER_B])
                else:
                    trained_role = Role.DEBATER_A
                opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A
                runtime = self._make_runtime(schedule)
                self._runtimes.append(runtime)
                envs.append(
                    DebateEnv(
                        role=trained_role,
                        runtime=runtime,
                        renderer=self.renderer,
                        opponent_completer=self.opponent_completer,
                        opponent_renderer=self.opponent_renderer,
                        opponent_role=opponent_role,
                    )
                )
            return envs

        # Normal (multi-agent / self-play) mode.
        # Validate all schedule actors have a corresponding env to prevent deadlock.
        required_roles = {r for slot in schedule for r in slot.actors}
        missing = required_roles - set(self.include_roles)
        if missing:
            raise ValueError(
                f"Schedule requires roles {missing} but include_roles={self.include_roles}. "
                f"All schedule actors must have an env, otherwise the runtime deadlocks."
            )

        envs: list[Env] = []
        self._runtimes = []
        for _ in range(self.group_size):
            runtime = self._make_runtime(schedule)
            self._runtimes.append(runtime)
            for role in self.include_roles:
                envs.append(DebateEnv(role=role, runtime=runtime, renderer=self.renderer))
        return envs

    def _make_runtime(self, schedule: tuple) -> DebateRuntime:
        prompts = resolve_prompts(self.game.prompts_ref)
        spec = DebateSpec(
            debate_id=uuid.uuid4().hex,
            problem=self.problem,
            schedule=schedule,
            think_visibility=prompts.get_think_visibility(),
            protocol_kind=self.game.protocol_kind,
            prompts_ref=self.game.prompts_ref,
        )
        state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=False,
            outcome=None,
        )
        return DebateRuntime(
            state,
            step_reward_fn=self.step_reward_fn,
            judge_callback=self.judge_callback,
        )

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        runtime_index_by_id: dict[int, int] = {}
        unique_runtimes: list[DebateRuntime] = []
        for env in env_group:
            assert isinstance(env, DebateEnv)
            runtime_id = id(env.runtime)
            if runtime_id in runtime_index_by_id:
                continue
            runtime_index_by_id[runtime_id] = len(unique_runtimes)
            unique_runtimes.append(env.runtime)

        states = [runtime.state for runtime in unique_runtimes]
        if self.metrics is not None and any(
            state.spec.problem.scoring_mode == ScoringMode.OPEN_ENDED for state in states
        ):
            raise ValueError(
                "Custom MetricFn injection is unsupported with OPEN_ENDED semantic scoring."
            )

        metrics_by_runtime = await _compute_metrics(
            states, self.metrics, self.scorer, self.scorer_parallelism
        )

        results = []
        for env in env_group:
            assert isinstance(env, DebateEnv)
            m = dict(metrics_by_runtime[runtime_index_by_id[id(env.runtime)]])
            if self.opponent_completer is not None:
                m = _remap_to_identity(m, env.role)
            outcome = env.runtime.state.outcome
            if outcome is None or self.outcome_reward_fn is None:
                results.append((0.0, m))
            else:
                rewards_by_role = self.outcome_reward_fn(outcome)
                results.append((rewards_by_role.get(env.role, 0.0), m))
        return results

    def advantage_subgroups(self, n_trajectories: int) -> tuple[tuple[int, ...], ...] | None:
        if self.opponent_completer is not None:
            return None  # frozen-opp: single side, default behavior
        return self.interleaved_subgroups(n_trajectories, len(self.include_roles))

    def logging_tags(self) -> list[str]:
        return ["debate", self.game.protocol_kind.value]


@dataclass
class DebateBranchGroupBuilder(EnvGroupBuilder):
    """Builds envs from a DebateSnapshot (for branching/MCTS)."""

    snapshot: DebateSnapshot
    renderer: Renderer
    step_reward_fn: StepRewardFn | None = None
    judge_callback: JudgeCallback | None = None
    outcome_reward_fn: OutcomeRewardFn | None = None
    include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B)
    metrics: dict[str, MetricFn] | None = field(default=None, repr=False)
    scorer: AnswerJudgeClient | None = field(default=None, repr=False)
    scorer_parallelism: int = 64

    _runtime: DebateRuntime | None = field(default=None, repr=False)

    async def make_envs(self) -> Sequence[Env]:
        forked = fork_state(self.snapshot.state)
        self._runtime = DebateRuntime(
            forked,
            step_reward_fn=self.step_reward_fn,
            judge_callback=self.judge_callback,
        )
        return [
            DebateEnv(role=role, runtime=self._runtime, renderer=self.renderer)
            for role in self.include_roles
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        if self._runtime is None:
            return [(0.0, {}) for _ in trajectory_group]
        state = self._runtime.state
        if self.metrics is not None and state.spec.problem.scoring_mode == ScoringMode.OPEN_ENDED:
            raise ValueError(
                "Custom MetricFn injection is unsupported with OPEN_ENDED semantic scoring."
            )
        m = (await _compute_metrics([state], self.metrics, self.scorer, self.scorer_parallelism))[0]
        if self.outcome_reward_fn is None:
            return [(0.0, m) for _ in trajectory_group]
        outcome = state.outcome
        if outcome is None:
            return [(0.0, m) for _ in trajectory_group]
        rewards_by_role = self.outcome_reward_fn(outcome)
        results = []
        for env in env_group:
            assert isinstance(env, DebateEnv)
            reward = rewards_by_role.get(env.role, 0.0)
            results.append((reward, m))
        return results

    def advantage_subgroups(self, n_trajectories: int) -> tuple[tuple[int, ...], ...] | None:
        # Branch builder is always self-play (no frozen-opponent variant)
        return self.interleaved_subgroups(n_trajectories, len(self.include_roles))

    def logging_tags(self) -> list[str]:
        return ["debate", "branch", self.snapshot.state.spec.protocol_kind.value]
