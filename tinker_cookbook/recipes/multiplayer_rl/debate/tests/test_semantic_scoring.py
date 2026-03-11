from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from tinker_cookbook.recipes.multiplayer_rl.debate.builders import DebateGroupBuilder
from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
    _STORE_KEY,
    _TRAINED_ROLE_KEY,
    _state_from_json,
    _state_to_json,
    debate_scorer,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.facts import (
    ResolvedDebateFacts,
    built_in_metric_values,
    resolve_debate_facts_for_states,
)
from tinker_cookbook.scoring import BinaryJudgeClient
from tinker_cookbook.scoring.types import normalize_binary_verdict_token
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    TurnSlot,
    Utterance,
    VisibilityPolicy,
)


def _slot(slot_id: int, round_index: int, phase: Phase, role: Role) -> TurnSlot:
    return TurnSlot(
        slot_id=slot_id,
        round_index=round_index,
        phase=phase,
        actors=(role,),
        boundary_after=False,
        visibility_policy=VisibilityPolicy.ALL_PRIOR,
    )


def _schedule_2_rounds() -> tuple[TurnSlot, ...]:
    return (
        _slot(0, 0, Phase.PROPOSE, Role.DEBATER_A),
        _slot(1, 0, Phase.PROPOSE, Role.DEBATER_B),
        _slot(2, 1, Phase.CRITIQUE, Role.DEBATER_A),
        _slot(3, 1, Phase.CRITIQUE, Role.DEBATER_B),
    )


def _utt(
    role: Role,
    round_index: int,
    phase: Phase,
    text: str,
    *,
    answer: str | None = None,
    slot_id: int = 0,
) -> Utterance:
    fields = {"answer": answer} if answer is not None else None
    return Utterance(
        role=role,
        round_index=round_index,
        phase=phase,
        text=text,
        token_count=len(text.split()),
        slot_id=slot_id,
        fields=fields,
    )


def _state(
    *,
    target: str | None,
    prompts_ref: str,
    scoring_mode: ScoringMode,
    transcript: tuple[Utterance, ...],
    winner: Role | None = Role.DEBATER_A,
) -> DebateState:
    outcome = None
    if winner is not ...:
        scores = {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
        if winner is not None:
            loser = Role.DEBATER_B if winner == Role.DEBATER_A else Role.DEBATER_A
            scores = {winner: 1.0, loser: -1.0}
        outcome = DebateOutcome(winner=winner, scores_by_role=scores)
    spec = DebateSpec(
        debate_id="debate-1",
        problem=DebateProblemSpec(
            task_prompt="What is the chemical formula for water?",
            scoring_mode=scoring_mode,
            answer_by_role=None,
            target=target,
        ),
        schedule=_schedule_2_rounds(),
        open_reasoning=False,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref=prompts_ref,
    )
    return DebateState(
        spec=spec,
        slot_index=len(transcript),
        rounds_completed=2,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=outcome,
    )


class _FakeJudgeClient(BinaryJudgeClient):
    def __init__(self, responses: dict[tuple[str, str], str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        key = (system, user)
        if key not in self.responses:
            raise AssertionError(f"Unexpected scorer prompt: {key!r}")
        return self.responses[key]


def test_binary_verdict_normalization_strips_trailing_punctuation():
    assert normalize_binary_verdict_token(" pass. ") == "PASS"
    assert normalize_binary_verdict_token("Different!!!") == "DIFFERENT"
    assert normalize_binary_verdict_token("") is None


def test_prompt_loader_exposes_binary_utility_templates():
    prompts = resolve_prompts(
        "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml"
    )

    matcher = prompts.get_binary_judge_template("matcher")
    grader = prompts.get_binary_judge_template("grader")

    assert matcher is not None
    assert matcher.positive == "SAME"
    assert matcher.negative == "DIFFERENT"
    assert grader is not None
    assert grader.positive == "PASS"
    assert grader.negative == "FAIL"


def test_prompt_loader_rejects_duplicate_binary_verdicts_after_normalization():
    with pytest.raises(ValueError, match="normalize to distinct verdicts"):
        resolve_prompts(
            "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/invalid_semantic_prompts.yaml"
        )


def test_scoring_mode_roundtrips_through_state_json():
    original = _state(
        target="A",
        prompts_ref="default",
        scoring_mode=ScoringMode.MCQ,
        transcript=(),
        winner=...,
    )

    restored = _state_from_json(_state_to_json(original))
    assert restored.spec.problem.scoring_mode == ScoringMode.MCQ


@pytest.mark.asyncio
async def test_resolve_debate_facts_open_ended_dedupes_calls_and_aligns_metrics():
    state = _state(
        target="water",
        prompts_ref="tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="water", slot_id=1),
            _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="H2O", slot_id=2),
            _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="water", slot_id=3),
        ),
        winner=Role.DEBATER_A,
    )
    prompts = resolve_prompts(state.spec.prompts_ref)
    matcher = prompts.get_binary_judge_template("matcher")
    grader = prompts.get_binary_judge_template("grader")
    assert matcher is not None and grader is not None

    _, matcher_user = matcher.render(question=state.spec.problem.task_prompt, a="H2O", b="water")
    _, grader_user = grader.render(
        question=state.spec.problem.task_prompt,
        target="water",
        response="H2O",
    )
    client = _FakeJudgeClient(
        {
            (matcher.system, matcher_user): "same.",
            (grader.system, grader_user): "PASS!",
        }
    )

    facts = await resolve_debate_facts_for_states(
        [state],
        scorer=client,
        prompts_for_ref=resolve_prompts,
    )

    assert len(facts) == 1
    metrics = built_in_metric_values(state, facts[0])
    assert metrics["accuracy.debater_a"] == 1.0
    assert metrics["accuracy.debater_b"] == 1.0
    assert metrics["truth_surfaced"] == 1.0
    assert metrics["disagreement"] == 0.0
    assert metrics["stance_change.debater_a"] == 0.0
    assert metrics["judge_quality"] == 1.0
    # One matcher call and one grader call. B-vs-target is exact-fast-pathed.
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_resolve_debate_facts_mcq_uses_zero_external_calls():
    state = _state(
        target="A",
        prompts_ref="default",
        scoring_mode=ScoringMode.MCQ,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="The answer is A", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="A", slot_id=1),
            _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="A", slot_id=2),
            _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="A", slot_id=3),
        ),
        winner=Role.DEBATER_A,
    )
    client = _FakeJudgeClient({})

    facts = await resolve_debate_facts_for_states(
        [state],
        scorer=client,
        prompts_for_ref=resolve_prompts,
    )

    assert isinstance(facts[0], ResolvedDebateFacts)
    metrics = built_in_metric_values(state, facts[0])
    assert metrics["accuracy.debater_a"] == 1.0
    assert metrics["accuracy.debater_b"] == 1.0
    assert metrics["disagreement"] == 0.0
    assert client.calls == []


class _PartialFailureClient(BinaryJudgeClient):
    """Client that succeeds on matcher calls but raises on grader calls."""

    def __init__(
        self,
        responses: dict[tuple[str, str], str],
        *,
        fail_keys: set[tuple[str, str]],
    ) -> None:
        self.responses = responses
        self.fail_keys = fail_keys
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        key = (system, user)
        if key in self.fail_keys:
            raise RuntimeError("synthetic grader failure")
        if key not in self.responses:
            raise AssertionError(f"Unexpected scorer prompt: {key!r}")
        return self.responses[key]


@pytest.mark.asyncio
async def test_open_ended_grader_error_produces_none_metrics():
    """Grader failure should yield None accuracy metrics, not crash."""
    state = _state(
        target="water",
        prompts_ref="tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="water", slot_id=1),
            _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="H2O", slot_id=2),
            _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="water", slot_id=3),
        ),
        winner=Role.DEBATER_A,
    )
    prompts = resolve_prompts(state.spec.prompts_ref)
    matcher = prompts.get_binary_judge_template("matcher")
    grader = prompts.get_binary_judge_template("grader")
    assert matcher is not None and grader is not None

    _, matcher_user = matcher.render(question=state.spec.problem.task_prompt, a="H2O", b="water")
    _, grader_user = grader.render(
        question=state.spec.problem.task_prompt,
        target="water",
        response="H2O",
    )
    # Matcher succeeds; grader raises.
    client = _PartialFailureClient(
        {(matcher.system, matcher_user): "same."},
        fail_keys={(grader.system, grader_user)},
    )

    facts = await resolve_debate_facts_for_states(
        [state],
        scorer=client,
        prompts_for_ref=resolve_prompts,
    )

    assert len(facts) == 1
    metrics = built_in_metric_values(state, facts[0])
    # Matcher succeeded: disagreement is resolved.
    assert metrics["disagreement"] == 0.0
    # Grader failed: accuracy should be None (error projected to missing key).
    assert metrics["accuracy.debater_a"] is None
    # B matched target exactly ("water" == "water") so gets exact-path correctness.
    assert metrics["accuracy.debater_b"] == 1.0


@pytest.mark.asyncio
async def test_open_ended_ambiguous_verdict_produces_none_metrics():
    """An ambiguous verdict (e.g. 'maybe') is absorbed by JudgeBatch and
    projected to None in metrics, not raised as an exception."""
    state = _state(
        target="water",
        prompts_ref="tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="water", slot_id=1),
        ),
        winner=Role.DEBATER_A,
    )
    prompts = resolve_prompts(state.spec.prompts_ref)
    matcher = prompts.get_binary_judge_template("matcher")
    assert matcher is not None
    _, matcher_user = matcher.render(question=state.spec.problem.task_prompt, a="H2O", b="water")
    client = _FakeJudgeClient({(matcher.system, matcher_user): "maybe"})

    facts = await resolve_debate_facts_for_states(
        [replace(state, outcome=None)],
        scorer=client,
        prompts_for_ref=resolve_prompts,
    )

    assert len(facts) == 1
    metrics = built_in_metric_values(state, facts[0])
    # Ambiguous matcher verdict → error → missing key → None disagreement.
    assert metrics["disagreement"] is None


def test_missing_grader_yaml_gets_default_grader():
    """A YAML without _grader still produces a grader via built-in default."""
    prompts = resolve_prompts(
        "tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/"
        "semantic_prompts_missing_grader.yaml"
    )
    grader = prompts.get_binary_judge_template("grader")
    assert grader is not None
    assert grader.positive == "CORRECT"
    assert grader.negative == "INCORRECT"


def _store_getter(json_str: str, trained_role: Role | None = None):
    def _get(key, default=None):
        if key == _STORE_KEY:
            return json_str
        if key == _TRAINED_ROLE_KEY:
            return trained_role.value if trained_role is not None else default
        return default

    return _get


@pytest.mark.asyncio
async def test_group_builder_open_ended_selfplay_scores_shared_runtime_once():
    state = _state(
        target="water",
        prompts_ref="tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="water", slot_id=1),
            _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="H2O", slot_id=2),
            _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="water", slot_id=3),
        ),
        winner=Role.DEBATER_A,
    )
    prompts = resolve_prompts(state.spec.prompts_ref)
    matcher = prompts.get_binary_judge_template("matcher")
    grader = prompts.get_binary_judge_template("grader")
    assert matcher is not None and grader is not None

    client = _FakeJudgeClient(
        {
            matcher.render(question=state.spec.problem.task_prompt, a="H2O", b="water"): "same.",
            grader.render(
                question=state.spec.problem.task_prompt,
                target="water",
                response="H2O",
            ): "PASS!",
        }
    )
    builder = DebateGroupBuilder(
        problem=DebateProblemSpec(
            task_prompt=state.spec.problem.task_prompt,
            scoring_mode=ScoringMode.OPEN_ENDED,
            answer_by_role=None,
            target="water",
        ),
        game=DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=2, prompts_ref=state.spec.prompts_ref),
        renderer=MagicMock(),
        scorer=client,
    )

    envs = await builder.make_envs()
    builder._runtimes[0]._state = state
    rewards = await builder.compute_group_rewards([MagicMock(), MagicMock()], envs)

    assert [reward for reward, _metrics in rewards] == [0.0, 0.0]
    for _reward, metrics in rewards:
        assert metrics["accuracy.debater_a"] == 1.0
        assert metrics["accuracy.debater_b"] == 1.0
        assert metrics["disagreement"] == 0.0
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_debate_scorer_open_ended_uses_semantic_facts():
    state = _state(
        target="water",
        prompts_ref="tinker_cookbook/recipes/multiplayer_rl/debate/tests/fixtures/semantic_prompts.yaml",
        scoring_mode=ScoringMode.OPEN_ENDED,
        transcript=(
            _utt(Role.DEBATER_A, 0, Phase.PROPOSE, "A", answer="H2O", slot_id=0),
            _utt(Role.DEBATER_B, 0, Phase.PROPOSE, "B", answer="water", slot_id=1),
            _utt(Role.DEBATER_A, 1, Phase.CRITIQUE, "A2", answer="H2O", slot_id=2),
            _utt(Role.DEBATER_B, 1, Phase.CRITIQUE, "B2", answer="water", slot_id=3),
        ),
        winner=Role.DEBATER_A,
    )
    prompts = resolve_prompts(state.spec.prompts_ref)
    matcher = prompts.get_binary_judge_template("matcher")
    grader = prompts.get_binary_judge_template("grader")
    assert matcher is not None and grader is not None

    client = _FakeJudgeClient(
        {
            matcher.render(question=state.spec.problem.task_prompt, a="H2O", b="water"): "SAME",
            grader.render(
                question=state.spec.problem.task_prompt,
                target="water",
                response="H2O",
            ): "pass",
        }
    )
    task_state = MagicMock(spec=TaskState)
    task_state.store = MagicMock()
    task_state.store.get = MagicMock(
        side_effect=_store_getter(_state_to_json(state), trained_role=Role.DEBATER_A)
    )

    score = await debate_scorer(scorer_client=client)(
        task_state,
        Target("water"),
    )

    assert score.value["accuracy.debater_a"] == 1.0
    assert score.value["accuracy.debater_b"] == 1.0
    assert score.value["id/accuracy.trained"] == 1.0
    assert score.value["id/accuracy.opponent"] == 1.0
    assert score.value["disagreement"] == 0.0
    assert len(client.calls) == 2
