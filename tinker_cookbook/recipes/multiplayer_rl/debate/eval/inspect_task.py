"""Inspect AI task, solver, and scorer for debate evaluation."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Mapping
from uuid import uuid4

from inspect_ai import Task, task
from inspect_ai.log import transcript
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import span

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import format_content_as_string

from ..core.reducer import get_eligible_roles
from ..core.schedule import build_schedule
from ..core.visibility import build_generation_messages
from ..scoring.judge import LLMJudgeCallback
from ..scoring.metrics import MetricFn, mcq_debate_metrics
from ..types import (
    DebateOutcome,
    DebateSpec,
    DebateState,
    JudgeDecision,
    Phase,
    ProtocolKind,
    Role,
    TurnSlot,
    Utterance,
    VisibilityPolicy,
)
from ..env import IDENTITY_REMAP_BASES, _remap_to_identity
from .dataset_adapter import DatasetAdapter

# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def _state_to_json(state: DebateState) -> str:
    return json.dumps(_encode_state(state))


def _state_from_json(s: str) -> DebateState:
    return _decode_state(json.loads(s))


def _encode_state(state: DebateState) -> dict[str, Any]:
    return {
        "spec": _encode_spec(state.spec),
        "slot_index": state.slot_index,
        "rounds_completed": state.rounds_completed,
        "transcript": [_encode_utterance(u) for u in state.transcript],
        "pending_simultaneous": {
            k.value: _encode_utterance(v) for k, v in state.pending_simultaneous.items()
        },
        "judge_trace": [_encode_judge_decision(d) for d in state.judge_trace],
        "done": state.done,
        "outcome": _encode_outcome(state.outcome) if state.outcome else None,
    }


def _encode_spec(spec: DebateSpec) -> dict[str, Any]:
    return {
        "debate_id": spec.debate_id,
        "task_prompt": spec.task_prompt,
        "answer_by_role": (
            {k.value: v for k, v in spec.answer_by_role.items()}
            if spec.answer_by_role is not None
            else None
        ),
        "schedule": [_encode_turn_slot(s) for s in spec.schedule],
        "open_reasoning": spec.open_reasoning,
        "protocol_kind": spec.protocol_kind.value,
        "prompts_ref": spec.prompts_ref,
        "target": spec.target,
    }


def _encode_turn_slot(slot: TurnSlot) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "round_index": slot.round_index,
        "phase": slot.phase.value,
        "actors": [r.value for r in slot.actors],
        "boundary_after": slot.boundary_after,
        "visibility_policy": slot.visibility_policy.value,
    }


def _encode_utterance(utt: Utterance) -> dict[str, Any]:
    return {
        "role": utt.role.value,
        "round_index": utt.round_index,
        "phase": utt.phase.value,
        "text": utt.text,
        "token_count": utt.token_count,
        "slot_id": utt.slot_id,
        "fields": dict(utt.fields) if utt.fields is not None else None,
    }


def _encode_judge_decision(d: JudgeDecision) -> dict[str, Any]:
    return {
        "round_index": d.round_index,
        "verdict": d.verdict,
        "score_delta_by_role": {k.value: v for k, v in d.score_delta_by_role.items()},
    }


def _encode_outcome(o: DebateOutcome) -> dict[str, Any]:
    return {
        "winner": o.winner.value if o.winner else None,
        "scores_by_role": {k.value: v for k, v in o.scores_by_role.items()},
        "verdict_text": o.verdict_text,
    }


# --- Decode ---


def _decode_state(d: dict[str, Any]) -> DebateState:
    return DebateState(
        spec=_decode_spec(d["spec"]),
        slot_index=d["slot_index"],
        rounds_completed=d["rounds_completed"],
        transcript=tuple(_decode_utterance(u) for u in d["transcript"]),
        pending_simultaneous={
            Role(k): _decode_utterance(v) for k, v in d["pending_simultaneous"].items()
        },
        judge_trace=tuple(_decode_judge_decision(j) for j in d["judge_trace"]),
        done=d["done"],
        outcome=_decode_outcome(d["outcome"]) if d["outcome"] else None,
    )


def _decode_spec(d: dict[str, Any]) -> DebateSpec:
    return DebateSpec(
        debate_id=d["debate_id"],
        task_prompt=d["task_prompt"],
        answer_by_role=(
            {Role(k): v for k, v in d["answer_by_role"].items()}
            if d["answer_by_role"] is not None
            else None
        ),
        schedule=tuple(_decode_turn_slot(s) for s in d["schedule"]),
        open_reasoning=d["open_reasoning"],
        protocol_kind=ProtocolKind(d["protocol_kind"]),
        prompts_ref=d["prompts_ref"],
        target=d["target"],
    )


def _decode_turn_slot(d: dict[str, Any]) -> TurnSlot:
    return TurnSlot(
        slot_id=d["slot_id"],
        round_index=d["round_index"],
        phase=Phase(d["phase"]),
        actors=tuple(Role(r) for r in d["actors"]),
        boundary_after=d["boundary_after"],
        visibility_policy=VisibilityPolicy(d["visibility_policy"]),
    )


def _decode_utterance(d: dict[str, Any]) -> Utterance:
    return Utterance(
        role=Role(d["role"]),
        round_index=d["round_index"],
        phase=Phase(d["phase"]),
        text=d["text"],
        token_count=d["token_count"],
        slot_id=d["slot_id"],
        fields=d["fields"],
    )


def _decode_judge_decision(d: dict[str, Any]) -> JudgeDecision:
    return JudgeDecision(
        round_index=d["round_index"],
        verdict=d["verdict"],
        score_delta_by_role={Role(k): v for k, v in d["score_delta_by_role"].items()},
    )


def _decode_outcome(d: dict[str, Any]) -> DebateOutcome:
    return DebateOutcome(
        winner=Role(d["winner"]) if d["winner"] else None,
        scores_by_role={Role(k): v for k, v in d["scores_by_role"].items()},
        verdict_text=d["verdict_text"],
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

_STORE_KEY = "debate_state_v1"
_TRAINED_ROLE_KEY = "trained_role_v1"


async def _drive_turn(
    runtime: Any,  # DebateRuntime — avoid circular at module scope
    role: Role,
    completer: MessageCompleter,
) -> bool:
    """Drive a single debater turn. Returns True if a turn was taken."""
    ticket = await runtime.wait_for_turn(role)
    if ticket is None:
        return False

    state: DebateState = runtime.state
    slot = state.spec.schedule[state.slot_index]
    async with span(f"{role.value}_{slot.phase.value}"):
        msgs, _prefill = build_generation_messages(state, role)

        # Log full input prompt.
        for i, msg in enumerate(msgs):
            transcript().info(f"[INPUT {role.value} msg {i}] role={msg['role']}\n{msg['content']}")

        response = await completer(msgs)
        text = format_content_as_string(response["content"], separator="")
        token_count = getattr(completer, "_last_output_tokens", len(text.split()) * 4 // 3)
        result = await runtime.submit(ticket, text, token_count)
        transcript().info(
            f"[OUTPUT {role.value}] round={slot.round_index}: "
            f"tokens~{token_count}, answer={result.logs.get('field.answer', '?')}"
        )
        transcript().info(text)
    return True


@solver
def debate_solver(
    sampling_client: MessageCompleter,
    opponent_client: MessageCompleter,
    judge_client: MessageCompleter,
    *,
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
    prompts_ref: str = "scientific_mcq",
    open_reasoning: bool = False,
    randomize_position: bool = True,
) -> Solver:
    """Solver that runs a full debate episode between two completers + judge."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from ..core.runtime import DebateRuntime

        # 1. Assign trained/opponent roles.
        if randomize_position:
            trained_role = random.choice([Role.DEBATER_A, Role.DEBATER_B])
        else:
            trained_role = Role.DEBATER_A
        opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A

        # Store trained_role for the scorer.
        state.store.set(_TRAINED_ROLE_KEY, trained_role.value)

        # 2. Build spec.
        debate_id = str(uuid4())
        schedule = build_schedule(protocol_kind, num_rounds)

        answer_a = state.metadata.get("answer_a", "")
        answer_b = state.metadata.get("answer_b", "")
        answer_by_role: Mapping[Role, str] | None = None
        if answer_a and answer_b:
            answer_by_role = {Role.DEBATER_A: answer_a, Role.DEBATER_B: answer_b}

        target_text: str | None = None
        if state.target is not None:
            target_text = state.target.text if hasattr(state.target, "text") else str(state.target)

        spec = DebateSpec(
            debate_id=debate_id,
            task_prompt=state.input_text,
            answer_by_role=answer_by_role,
            schedule=schedule,
            open_reasoning=open_reasoning,
            protocol_kind=protocol_kind,
            prompts_ref=prompts_ref,
            target=target_text,
        )

        initial_state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=False,
            outcome=None,
        )

        # 3. Create runtime with judge callback.
        judge_cb = LLMJudgeCallback(judge_client)
        runtime = DebateRuntime(initial_state, judge_callback=judge_cb)

        completers = {
            trained_role: sampling_client,
            opponent_role: opponent_client,
        }

        # 4. Drive episode.
        async with span("debate_episode"):
            while not runtime.state.done:
                eligible = get_eligible_roles(runtime.state)
                debater_roles = [r for r in eligible if r != Role.JUDGE]
                if not debater_roles:
                    break

                if len(debater_roles) > 1:
                    # Simultaneous slot: drive both concurrently so the barrier
                    # in runtime.submit can resolve.
                    await asyncio.gather(
                        *[_drive_turn(runtime, r, completers[r]) for r in debater_roles]
                    )
                else:
                    await _drive_turn(runtime, debater_roles[0], completers[debater_roles[0]])

        # 5. Persist debate state for the scorer.
        state.store.set(_STORE_KEY, _state_to_json(runtime.state))

        # 6. Set output.
        verdict = ""
        if runtime.state.outcome and runtime.state.outcome.verdict_text:
            verdict = runtime.state.outcome.verdict_text
        state.output = ModelOutput.from_content(model="debate", content=verdict)

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _identity_metric_keys() -> list[str]:
    """Compute all possible id/ metric keys from IDENTITY_REMAP_BASES."""
    keys = ["id/trained_role_is_a"]
    for base in IDENTITY_REMAP_BASES:
        keys.append(f"id/{base}.trained")
        keys.append(f"id/{base}.opponent")
    return keys


def debate_scorer(
    metrics: dict[str, MetricFn] | None = None,
) -> Scorer:
    """Scorer that evaluates debate outcomes via MetricFn functions.

    Returns a dict-valued Score where each key is a metric name and each value
    is the metric result. Per-key ``mean()`` aggregation is applied via the
    ``metrics`` parameter of the ``@scorer`` decorator.
    """
    resolved = metrics if metrics is not None else mcq_debate_metrics()
    # Build per-key metrics dict for @scorer: {metric_name: [mean()]}
    _metric_aggs = {name: [mean()] for name in resolved}
    # Add identity-remapped metric keys for aggregation.
    for key in _identity_metric_keys():
        _metric_aggs[key] = [mean()]

    @scorer(metrics=_metric_aggs)
    def _scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            raw = state.store.get(_STORE_KEY, None)
            if raw is None:
                # Fill all registered keys with NaN so aggregation doesn't
                # fail on missing keys.
                nan = float("nan")
                empty_values = {name: nan for name in resolved}
                for key in _identity_metric_keys():
                    empty_values[key] = nan
                return Score(
                    value=empty_values,
                    explanation="No debate state found in store",
                )

            debate_state = _state_from_json(raw)
            nan = float("nan")
            values: dict[str, float] = {}
            for name, fn in resolved.items():
                result = fn(debate_state)
                values[name] = result.value if result.value is not None else nan

            # Identity remap: translate seat-based to trained/opponent metrics.
            trained_role_str = state.store.get(_TRAINED_ROLE_KEY, None)
            if trained_role_str is not None:
                trained_role = Role(trained_role_str)
                values = _remap_to_identity(values, trained_role)
            else:
                # Fill identity keys with NaN so _metric_aggs aggregation
                # doesn't fail on missing keys.
                for key in _identity_metric_keys():
                    values[key] = nan

            return Score(
                value=values,
                answer=_winner_str(debate_state),
                explanation=debate_state.outcome.verdict_text if debate_state.outcome else None,
            )

        return score

    return _scorer()


def _winner_str(state: DebateState) -> str:
    if state.outcome is None:
        return "no_outcome"
    if state.outcome.winner is None:
        return "tie"
    return state.outcome.winner.value


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@task
def debate_eval(
    adapter: DatasetAdapter,
    sampling_client: MessageCompleter,
    opponent_client: MessageCompleter,
    judge_client: MessageCompleter,
    *,
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
    prompts_ref: str = "scientific_mcq",
    open_reasoning: bool = False,
    randomize_position: bool = True,
) -> Task:
    """Inspect AI task that runs debate eval end-to-end."""
    return Task(
        dataset=adapter.to_samples(),
        solver=debate_solver(
            sampling_client,
            opponent_client,
            judge_client,
            protocol_kind=protocol_kind,
            num_rounds=num_rounds,
            prompts_ref=prompts_ref,
            open_reasoning=open_reasoning,
            randomize_position=randomize_position,
        ),
        scorer=debate_scorer(),
    )
