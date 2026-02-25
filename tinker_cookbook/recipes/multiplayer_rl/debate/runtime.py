"""Async coordination shell over the pure reducer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from tinker_cookbook.renderers import Message
from tinker_cookbook.utils import logtree

from .parsing import extract_fields
from .plugins import JudgeCallback, StepRewardFn
from .prompts import resolve_prompts
from .types import Phase

# Map schedule phase values to YAML trigger keys where they differ.
_PHASE_TO_TRIGGER: dict[str, str] = {
    Phase.JUDGE_VERDICT.value: "final",
    Phase.JUDGE_QUERY.value: "boundary",
}
from .reducer import apply_action, apply_judge_event, get_current_slot, get_eligible_roles
from .types import (
    DebateSnapshot,
    DebateState,
    JudgeRequest,
    ProtocolKind,
    Role,
    TurnTicket,
)
from .visibility import get_visible_messages

# Type aliases matching tinker_cookbook.rl.types
StopCondition = list[str] | list[int]
Metrics = dict[str, float | int]
Logs = dict[str, str | int | float]


@dataclass
class SubmitResult:
    reward: float
    episode_done: bool
    messages: tuple[Message, ...]
    stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)


class DebateRuntime:
    def __init__(
        self,
        initial_state: DebateState,
        step_reward_fn: StepRewardFn | None = None,
        judge_callback: JudgeCallback | None = None,
    ) -> None:
        self._state = initial_state
        self._step_reward_fn = step_reward_fn
        self._judge_callback = judge_callback
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    @property
    def state(self) -> DebateState:
        return self._state

    def snapshot(
        self,
        renderer_name: str,
        protocol_kind: ProtocolKind,
        protocol_kwargs: Mapping[str, Any],
    ) -> DebateSnapshot:
        return DebateSnapshot(
            state=self._state,
            protocol_kind=protocol_kind,
            protocol_kwargs=protocol_kwargs,
            renderer_name=renderer_name,
        )

    def _get_messages(self, state: DebateState, role: Role) -> tuple[Message, ...]:
        return get_visible_messages(state, role)

    async def wait_for_turn(self, role: Role) -> TurnTicket | None:
        """Block until role is eligible or episode is done. Return ticket or None."""
        async with self._condition:
            try:
                while True:
                    if self._state.done:
                        return None
                    if role in get_eligible_roles(self._state):
                        slot = get_current_slot(self._state)
                        assert slot is not None
                        return TurnTicket(
                            slot_id=slot.slot_id,
                            state_version=self._state.version,
                            role=role,
                        )
                    await self._condition.wait()
            except asyncio.CancelledError:
                self._condition.notify_all()
                raise

    async def submit(
        self, ticket: TurnTicket, text: str, token_count: int
    ) -> SubmitResult:
        """Validate ticket, apply action, handle boundary/final callbacks.

        For simultaneous slots, the first arriver buffers and waits on the
        condition until the last arriver commits. Both return consistent results.

        Judge callbacks are awaited while holding the lock. This is intentional:
        releasing the lock would let next-turn actors proceed before the judge
        has processed the boundary, which is semantically incorrect. Callbacks
        receive a state snapshot via JudgeRequest and must not call back into
        the runtime.
        """
        async with self._condition:
            # Stale ticket check: slot_id must match current slot.
            # (Version check alone fails for simultaneous slots where buffering
            # changes version between the first and second arriver.)
            current_slot = get_current_slot(self._state)
            if current_slot is None or ticket.slot_id != current_slot.slot_id:
                raise ValueError(
                    f"Stale ticket: ticket slot {ticket.slot_id}, "
                    f"current slot {current_slot.slot_id if current_slot else 'None (exhausted)'}"
                )

            # Capture slot info before action (slot advances after apply_action).
            slot_phase = current_slot.phase.value
            slot_round = current_slot.round_index

            # Extract structured fields if specs exist for this role/phase.
            fields = None
            prompts = resolve_prompts(self._state.spec.prompts_ref)
            trigger = _PHASE_TO_TRIGGER.get(slot_phase, slot_phase)
            field_specs = prompts.get_field_specs(ticket.role.value, trigger)
            if field_specs:
                fields = extract_fields(text, field_specs)

            before = self._state
            result = apply_action(self._state, ticket.role, text, token_count, fields=fields)
            self._state = result.new_state

            # Per-submitter logs: debate context for this action.
            base_logs: Logs = {
                "role": ticket.role.value,
                "phase": slot_phase,
                "round": slot_round,
                "text": text,
                "output_tokens": token_count,
            }
            if fields is not None:
                for k, v in fields.items():
                    base_logs[f"field.{k}"] = v
            elif field_specs:
                base_logs["field_extraction_failed"] = 1

            if not result.committed:
                # Simultaneous slot: we buffered. Wait for the last arriver to commit.
                self._condition.notify_all()
                slot_before = ticket.slot_id
                try:
                    while True:
                        slot = get_current_slot(self._state)
                        # Slot advanced (committed) or episode done.
                        if slot is None or slot.slot_id != slot_before or self._state.done:
                            break
                        await self._condition.wait()
                except asyncio.CancelledError:
                    # Roll back our pending utterance to prevent ghost actions.
                    # If the slot hasn't advanced, remove our buffered entry.
                    current = get_current_slot(self._state)
                    if current is not None and current.slot_id == slot_before:
                        rolled_back = {
                            k: v for k, v in self._state.pending_simultaneous.items()
                            if k != ticket.role
                        }
                        self._state = replace(self._state, pending_simultaneous=rolled_back)
                    self._condition.notify_all()
                    raise

                # Now state reflects the commit. Compute reward post-callback
                # (same timing as committer path — see below).
                reward = 0.0
                if self._step_reward_fn is not None:
                    utt = next(
                        (u for u in reversed(self._state.transcript) if u.role == ticket.role),
                        None,
                    )
                    reward = self._step_reward_fn(before, self._state, ticket.role, utt)

                messages = self._get_messages(self._state, ticket.role)
                return SubmitResult(
                    reward=reward,
                    episode_done=self._state.done,
                    messages=messages,
                    stop_condition=[],
                    logs=base_logs,
                )

            # We committed (sequential slot, or last arriver in simultaneous).
            # try/finally ensures notify_all fires even if reward/callback raises,
            # preventing lost wakeups for barrier waiters.
            try:
                # Log committed utterances to logtree trace.
                for utt in result.committed:
                    answer = utt.fields.get("answer") if utt.fields else None
                    fields_tag = f" answer={answer}" if answer else ""
                    logtree.log_text(
                        f"Round {utt.round_index} [{utt.phase.value}] "
                        f"{utt.role.value}:{fields_tag} {utt.text[:200]}  "
                        f"[{utt.token_count} tokens out]"
                    )

                # Handle boundary: judge callback (before reward computation).
                if result.boundary_reached and self._judge_callback is not None:
                    request = JudgeRequest(state=self._state, trigger="boundary")
                    decision = await self._judge_callback.on_boundary(request)
                    if decision is not None:
                        self._state = apply_judge_event(self._state, decision)

                # Handle episode end: judge final callback.
                if result.episode_done and self._judge_callback is not None:
                    request = JudgeRequest(state=self._state, trigger="final")
                    outcome = await self._judge_callback.on_final(request)
                    self._state = replace(self._state, outcome=outcome)
                    # Log judge verdict.
                    winner_str = outcome.winner.value if outcome.winner else "tie"
                    logtree.log_text(f"Judge verdict: {winner_str}")
                    if outcome.verdict_text:
                        logtree.details(
                            outcome.verdict_text,
                            summary=f"Judge reasoning ({len(outcome.verdict_text)} chars)",
                            pre=True,
                        )

                # Compute reward post-callback so both arrivers see the same state.
                reward = 0.0
                if self._step_reward_fn is not None:
                    utt = next((u for u in result.committed if u.role == ticket.role), None)
                    reward = self._step_reward_fn(before, self._state, ticket.role, utt)

                messages = self._get_messages(self._state, ticket.role)
            finally:
                # Always notify — prevents lost wakeups if reward_fn or callback raises.
                self._condition.notify_all()

        return SubmitResult(
            reward=reward,
            episode_done=result.episode_done,
            messages=messages,
            stop_condition=[],
            logs=base_logs,
        )
