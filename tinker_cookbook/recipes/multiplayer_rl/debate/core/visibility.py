"""Visibility policies and message assembly for debate environments."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

from tinker_cookbook.renderers import Message

from ..prompts import resolve_prompts
from ..types import DebateState, Role, TurnSlot, Utterance, VisibilityPolicy, current_phase

VisibilityFn = Callable[[DebateState, Role], list[Utterance]]
REGISTRY: dict[VisibilityPolicy, VisibilityFn] = {}


def register(policy: VisibilityPolicy) -> Callable[[VisibilityFn], VisibilityFn]:
    """Decorator to register a visibility policy."""

    def decorator(fn: VisibilityFn) -> VisibilityFn:
        REGISTRY[policy] = fn
        return fn

    return decorator


_ROLE_LABELS: dict[Role, str] = {
    Role.DEBATER_A: "A",
    Role.DEBATER_B: "B",
    Role.JUDGE: "Judge",
}


def _system_message(state: DebateState, viewer: Role) -> Message:
    prompts = resolve_prompts(state.spec.prompts_ref)
    return Message(role="system", content=prompts.render_system(state, viewer))


def _wrap_opponent_turn(utt: Utterance, open_reasoning: bool, prompts=None, viewer=None) -> str:
    label = _ROLE_LABELS[utt.role]
    text = utt.text if open_reasoning else utt.stripped_text
    if prompts is not None and prompts.opponent_wrap is not None and viewer is not None:
        return prompts.render_opponent_wrap(text, label, utt.phase.value, viewer)
    return f'<opponent_turn agent="{label}" phase="{utt.phase.value}">\n{text}\n</opponent_turn>'


def _utterance_to_message(utt: Utterance, viewer: Role, open_reasoning: bool, prompts=None) -> Message:
    """Convert utterance to Message. Own turns -> assistant (full text), opponent -> user (wrapped)."""
    if utt.role == viewer:
        # Keep full text (including thinking) for KV-cache prefix reuse.
        return Message(role="assistant", content=utt.text)
    else:
        return Message(role="user", content=_wrap_opponent_turn(utt, open_reasoning, prompts=prompts, viewer=viewer))


def _shuffle_simultaneous(
    utterances: list[Utterance],
    viewer: Role,
    state: DebateState,
) -> list[Utterance]:
    """Shuffle utterance order within simultaneous slots for judge debiasing."""
    if viewer != Role.JUDGE:
        return utterances
    debate_id = state.spec.debate_id

    # Build slot_id -> list of utterances, preserving first-seen order
    slot_groups: dict[int, list[Utterance]] = {}
    order: list[int] = []
    for utt in utterances:
        if utt.slot_id not in slot_groups:
            slot_groups[utt.slot_id] = []
            order.append(utt.slot_id)
        slot_groups[utt.slot_id].append(utt)

    # Find which slots are simultaneous (len(actors) > 1)
    schedule = state.spec.schedule
    slot_map: dict[int, TurnSlot] = {s.slot_id: s for s in schedule}

    result: list[Utterance] = []
    for sid in order:
        group = slot_groups[sid]
        slot_def = slot_map.get(sid)
        if slot_def is not None and len(slot_def.actors) > 1 and len(group) > 1:
            # Deterministic shuffle seeded by debate_id + slot_id
            seed = int(hashlib.sha256(f"{debate_id}|{sid}".encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            rng.shuffle(group)
        result.extend(group)
    return result


def _consolidate_str_messages(msgs: list[Message]) -> list[Message]:
    """Merge adjacent same-role messages. Type-gated: only str content, skip system."""
    if not msgs:
        return msgs
    result: list[Message] = []
    for msg in msgs:
        if (
            result
            and isinstance(msg.get("content"), str)
            and isinstance(result[-1].get("content"), str)
            and msg["role"] == result[-1]["role"]
            and msg["role"] != "system"
        ):
            result[-1] = Message(
                role=msg["role"],
                content=result[-1]["content"] + "\n\n" + msg["content"],
            )
        else:
            result.append(msg)
    return result


def _question_message(state: DebateState, viewer: Role) -> Message | None:
    prompts = resolve_prompts(state.spec.prompts_ref)
    q = prompts.render_question(state, viewer)
    if q is None:
        return None
    return Message(role="user", content=q)


@register(VisibilityPolicy.ALL_PRIOR)
def all_prior(state: DebateState, viewer: Role) -> list[Utterance]:
    """All committed utterances."""
    return list(state.transcript)


@register(VisibilityPolicy.COMPLETED_ROUNDS_ONLY)
def completed_rounds_only(state: DebateState, viewer: Role) -> list[Utterance]:
    """Only utterances from completed rounds. For simultaneous mid-round hiding."""
    return [u for u in state.transcript if u.round_index < state.rounds_completed]


def get_visible_messages(state: DebateState, viewer: Role) -> tuple[Message, ...]:
    """Observation messages: system + question + transcript (no instructions)."""
    prompts = resolve_prompts(state.spec.prompts_ref)
    msgs: list[Message] = [_system_message(state, viewer)]

    q = _question_message(state, viewer)
    if q is not None:
        msgs.append(q)

    # Get visible utterances via policy
    schedule = state.spec.schedule
    if state.slot_index < len(schedule):
        policy_name = schedule[state.slot_index].visibility_policy
    else:
        policy_name = VisibilityPolicy.ALL_PRIOR

    fn = REGISTRY[policy_name]
    utterances = fn(state, viewer)

    # Shuffle simultaneous slots for judge
    utterances = _shuffle_simultaneous(utterances, viewer, state)

    # Convert to Messages
    for utt in utterances:
        msgs.append(_utterance_to_message(utt, viewer, state.spec.open_reasoning, prompts=prompts))

    return tuple(msgs)


def build_generation_messages(
    state: DebateState,
    viewer: Role,
    *,
    trigger: str | None = None,
) -> tuple[list[Message], str | None]:
    """Full prompt assembly: observation + interleaved instructions + prefill.

    Produces a prefix-monotonic message sequence for KV-cache reuse:
      [system] [question] [instr_1] [own_turn_1] [opp_turn_1] [instr_2] [own_turn_2] ... [current_instr]

    Historical phase instructions are reconstructed before each own-turn.
    Adjacent same-role user messages are consolidated for API compliance.

    Returns (messages, prefill_string_or_None).
    """
    prompts = resolve_prompts(state.spec.prompts_ref)

    # 1. Fixed prefix: system + question
    msgs: list[Message] = [_system_message(state, viewer)]
    q = _question_message(state, viewer)
    if q is not None:
        msgs.append(q)

    # 2. Get visible utterances (policy-filtered, shuffled for judge)
    schedule = state.spec.schedule
    if state.slot_index < len(schedule):
        policy_name = schedule[state.slot_index].visibility_policy
    else:
        policy_name = VisibilityPolicy.ALL_PRIOR

    utterances = REGISTRY[policy_name](state, viewer)
    utterances = _shuffle_simultaneous(utterances, viewer, state)

    # 3. Build transcript with interleaved phase instructions before own turns
    transcript_msgs: list[Message] = []
    for utt in utterances:
        if utt.role == viewer:
            # Insert reconstructed phase instruction before own turn
            instr = prompts.render_user(state, viewer, trigger=utt.phase.value)
            if instr:
                transcript_msgs.append(Message(role="user", content=instr))
        transcript_msgs.append(
            _utterance_to_message(utt, viewer, state.spec.open_reasoning, prompts=prompts)
        )

    # 4. Consolidate transcript (merges adjacent same-role user messages)
    msgs.extend(_consolidate_str_messages(transcript_msgs))

    # 5. Resolve trigger and append current instruction
    if trigger is None:
        trigger = current_phase(state)

    user_content = prompts.render_user(state, viewer, trigger=trigger)
    if user_content:
        msgs.append(Message(role="user", content=user_content))

    # 6. Resolve prefill
    prefill = prompts.render_prefill(state, viewer, trigger=trigger)

    return msgs, prefill
