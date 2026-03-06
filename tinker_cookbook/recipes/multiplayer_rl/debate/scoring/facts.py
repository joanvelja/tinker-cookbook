"""Post-episode semantic fact resolution for debate metrics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from ..prompts import BinaryJudgeTemplate, DebatePrompts, normalize_binary_verdict_token
from ..types import DebateState, Phase, Role, ScoringMode
from .mcq import normalize_mcq
from .metrics import (
    _latest_think_answer,
    parse_success,
    parse_success_by_role,
    think_answer_parse_rate,
    think_block_rate,
)
from .providers import AnswerJudgeClient
from .trajectory import answers_by_round, final_answer

MatcherKey = tuple[str, str, str, str]
GraderKey = tuple[str, str, str, str]


class BinaryJudgeError(ValueError):
    """Raised when a binary judge returns an invalid verdict."""


@dataclass(frozen=True)
class ResolvedDebateFacts:
    scoring_mode: ScoringMode
    equivalence: Mapping[MatcherKey, bool]
    correctness: Mapping[GraderKey, bool]
    telemetry: Mapping[str, int | float | str]


@dataclass(frozen=True)
class _StatePlan:
    scoring_mode: ScoringMode
    equivalence_keys: frozenset[MatcherKey]
    correctness_keys: frozenset[GraderKey]


def exact_match(predicted: str, target: str) -> bool:
    return predicted.strip().lower() == target.strip().lower()


def _mcq_match(predicted: str, target: str) -> bool:
    predicted_norm = normalize_mcq(predicted)
    target_norm = normalize_mcq(target)
    if predicted_norm is not None and target_norm is not None:
        return predicted_norm == target_norm
    return exact_match(predicted, target)


def _normalized_cache_text(text: str) -> str:
    return text.strip()


def _matcher_key(scoring_mode: ScoringMode, question: str, left: str, right: str) -> MatcherKey:
    left_norm = _normalized_cache_text(left)
    right_norm = _normalized_cache_text(right)
    ordered = tuple(sorted((left_norm, right_norm)))
    return (scoring_mode.value, question, ordered[0], ordered[1])


def _grader_key(
    scoring_mode: ScoringMode,
    question: str,
    target: str,
    response: str,
) -> GraderKey:
    return (
        scoring_mode.value,
        question,
        _normalized_cache_text(target),
        _normalized_cache_text(response),
    )


async def _binary_judge(
    scorer: AnswerJudgeClient,
    template: BinaryJudgeTemplate,
    *,
    judge_kind: str,
    render_kwargs: dict[str, str],
) -> bool:
    user = template.user.render(**render_kwargs)
    response_text = await scorer.complete_binary(
        system=template.system,
        user=user,
        kind=judge_kind,
    )
    verdict = normalize_binary_verdict_token(response_text)
    if verdict == template.positive:
        return True
    if verdict == template.negative:
        return False
    raise BinaryJudgeError(
        f"Unrecognized verdict: {response_text!r}. "
        f"Expected {template.positive!r} or {template.negative!r}."
    )


def _require_binary_template(
    prompts: DebatePrompts,
    name: str,
) -> BinaryJudgeTemplate:
    template = prompts.get_binary_judge_template(name)  # type: ignore[arg-type]
    if template is None:
        raise ValueError(
            f"Open-ended scoring requires _{name} in {prompts.source_ref}, but it is missing."
        )
    return template


def _freeze_mapping(mapping: dict) -> Mapping:
    return MappingProxyType(dict(mapping))


def _state_answers(state: DebateState, role: Role) -> list[str]:
    return [answer for answer in answers_by_round(state, role=role) if answer is not None]


async def resolve_debate_facts_for_states(
    states: list[DebateState],
    *,
    scorer: AnswerJudgeClient | None,
    prompts_for_ref: Callable[[str], DebatePrompts],
    parallelism: int,
) -> list[ResolvedDebateFacts]:
    semaphore = asyncio.Semaphore(max(1, parallelism))
    equivalence_results: dict[MatcherKey, bool] = {}
    correctness_results: dict[GraderKey, bool] = {}
    equivalence_tasks: dict[MatcherKey, asyncio.Task[bool]] = {}
    correctness_tasks: dict[GraderKey, asyncio.Task[bool]] = {}
    prompt_cache: dict[str, DebatePrompts] = {}
    telemetry = {"llm_calls": 0, "cache_hits": 0, "cache_misses": 0}
    plans: list[_StatePlan] = []

    def _prompts(ref: str) -> DebatePrompts:
        if ref not in prompt_cache:
            prompt_cache[ref] = prompts_for_ref(ref)
        return prompt_cache[ref]

    def _store_result(mapping: dict, key: tuple, value: bool) -> None:
        existing = mapping.get(key)
        if existing is None:
            mapping[key] = value

    def _schedule_equivalence(
        state: DebateState,
        left: str,
        right: str,
        *,
        question: str,
        prompts: DebatePrompts | None,
    ) -> MatcherKey:
        key = _matcher_key(state.spec.scoring_mode, question, left, right)
        if key in equivalence_results or key in equivalence_tasks:
            telemetry["cache_hits"] += 1
            return key

        telemetry["cache_misses"] += 1
        if state.spec.scoring_mode == ScoringMode.MCQ:
            _store_result(equivalence_results, key, _mcq_match(left, right))
            return key

        if exact_match(left, right):
            _store_result(equivalence_results, key, True)
            return key

        if scorer is None:
            raise ValueError("Open-ended scoring requires a configured scorer client.")
        assert prompts is not None
        template = _require_binary_template(prompts, "matcher")
        ordered_left, ordered_right = sorted((left, right), key=lambda text: text.strip())

        async def _run() -> bool:
            async with semaphore:
                telemetry["llm_calls"] += 1
                return await _binary_judge(
                    scorer,
                    template,
                    judge_kind="matcher",
                    render_kwargs={
                        "question": question,
                        "a": ordered_left,
                        "b": ordered_right,
                    },
                )

        equivalence_tasks[key] = asyncio.create_task(_run())
        return key

    def _schedule_correctness(
        state: DebateState,
        response: str,
        target: str,
        *,
        question: str,
        prompts: DebatePrompts | None,
    ) -> GraderKey:
        key = _grader_key(state.spec.scoring_mode, question, target, response)
        if key in correctness_results or key in correctness_tasks:
            telemetry["cache_hits"] += 1
            return key

        telemetry["cache_misses"] += 1
        if state.spec.scoring_mode == ScoringMode.MCQ:
            _store_result(correctness_results, key, _mcq_match(response, target))
            return key

        if exact_match(response, target):
            _store_result(correctness_results, key, True)
            return key

        if scorer is None:
            raise ValueError("Open-ended scoring requires a configured scorer client.")
        assert prompts is not None
        template = _require_binary_template(prompts, "grader")

        async def _run() -> bool:
            async with semaphore:
                telemetry["llm_calls"] += 1
                return await _binary_judge(
                    scorer,
                    template,
                    judge_kind="grader",
                    render_kwargs={
                        "question": question,
                        "target": target,
                        "response": response,
                    },
                )

        correctness_tasks[key] = asyncio.create_task(_run())
        return key

    for state in states:
        question = state.spec.task_prompt
        prompts = _prompts(state.spec.prompts_ref) if state.spec.scoring_mode == ScoringMode.OPEN_ENDED else None
        equivalence_keys: set[MatcherKey] = set()
        correctness_keys: set[GraderKey] = set()

        final_a = final_answer(state, role=Role.DEBATER_A)
        final_b = final_answer(state, role=Role.DEBATER_B)
        if final_a is not None and final_b is not None:
            equivalence_keys.add(
                _schedule_equivalence(state, final_a, final_b, question=question, prompts=prompts)
            )

        for role in (Role.DEBATER_A, Role.DEBATER_B):
            answers = _state_answers(state, role)
            if len(answers) >= 2:
                first = answers[0]
                for later in answers[1:]:
                    equivalence_keys.add(
                        _schedule_equivalence(
                            state,
                            first,
                            later,
                            question=question,
                            prompts=prompts,
                        )
                    )

        rounds_a = answers_by_round(state, role=Role.DEBATER_A)
        rounds_b = answers_by_round(state, role=Role.DEBATER_B)
        for answer_a, answer_b in zip(rounds_a, rounds_b, strict=False):
            if answer_a is None or answer_b is None:
                continue
            equivalence_keys.add(
                _schedule_equivalence(
                    state,
                    answer_a,
                    answer_b,
                    question=question,
                    prompts=prompts,
                )
            )

        if state.spec.target is not None:
            target = state.spec.target
            for answer in filter(None, [final_a, final_b]):
                correctness_keys.add(
                    _schedule_correctness(
                        state,
                        answer,
                        target,
                        question=question,
                        prompts=prompts,
                    )
                )
            for role in (Role.DEBATER_A, Role.DEBATER_B):
                for answer in _state_answers(state, role):
                    correctness_keys.add(
                        _schedule_correctness(
                            state,
                            answer,
                            target,
                            question=question,
                            prompts=prompts,
                        )
                    )
                think_answer = _latest_think_answer(state, role)
                if think_answer is not None:
                    correctness_keys.add(
                        _schedule_correctness(
                            state,
                            think_answer,
                            target,
                            question=question,
                            prompts=prompts,
                        )
                    )

        for role in (Role.DEBATER_A, Role.DEBATER_B):
            think_answer = _latest_think_answer(state, role)
            public_answer = final_answer(state, role=role)
            if think_answer is not None and public_answer is not None:
                equivalence_keys.add(
                    _schedule_equivalence(
                        state,
                        think_answer,
                        public_answer,
                        question=question,
                        prompts=prompts,
                    )
                )

        plans.append(
            _StatePlan(
                scoring_mode=state.spec.scoring_mode,
                equivalence_keys=frozenset(equivalence_keys),
                correctness_keys=frozenset(correctness_keys),
            )
        )

    async def _collect(
        tasks: dict[tuple[str, ...], asyncio.Task[bool]],
        sink: dict[tuple[str, ...], bool],
    ) -> None:
        if not tasks:
            return
        try:
            completed = await asyncio.gather(*tasks.values())
        except Exception:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        for key, value in zip(tasks, completed, strict=True):
            sink[key] = value

    try:
        await _collect(equivalence_tasks, equivalence_results)
        await _collect(correctness_tasks, correctness_results)
    except Exception:
        outstanding = [*equivalence_tasks.values(), *correctness_tasks.values()]
        for task in outstanding:
            if not task.done():
                task.cancel()
        if outstanding:
            await asyncio.gather(*outstanding, return_exceptions=True)
        raise

    resolved: list[ResolvedDebateFacts] = []
    for plan in plans:
        resolved.append(
            ResolvedDebateFacts(
                scoring_mode=plan.scoring_mode,
                equivalence=_freeze_mapping(
                    {key: equivalence_results[key] for key in plan.equivalence_keys}
                ),
                correctness=_freeze_mapping(
                    {key: correctness_results[key] for key in plan.correctness_keys}
                ),
                telemetry=_freeze_mapping(dict(telemetry)),
            )
        )
    return resolved


def _lookup_equivalence(
    facts: ResolvedDebateFacts,
    state: DebateState,
    left: str | None,
    right: str | None,
) -> bool | None:
    if left is None or right is None:
        return None
    key = _matcher_key(facts.scoring_mode, state.spec.task_prompt, left, right)
    return facts.equivalence.get(key)


def _lookup_correctness(
    facts: ResolvedDebateFacts,
    state: DebateState,
    response: str | None,
) -> bool | None:
    if response is None or state.spec.target is None:
        return None
    key = _grader_key(facts.scoring_mode, state.spec.task_prompt, state.spec.target, response)
    return facts.correctness.get(key)


def _float(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


def built_in_metric_values(
    state: DebateState,
    facts: ResolvedDebateFacts,
) -> dict[str, float | None]:
    final_a = final_answer(state, role=Role.DEBATER_A)
    final_b = final_answer(state, role=Role.DEBATER_B)
    first_last_by_role = {role: _state_answers(state, role) for role in (Role.DEBATER_A, Role.DEBATER_B)}

    a_correct = _lookup_correctness(facts, state, final_a)
    b_correct = _lookup_correctness(facts, state, final_b)
    final_same = _lookup_equivalence(facts, state, final_a, final_b)

    values: dict[str, float | None] = {
        "accuracy.debater_a": _float(a_correct),
        "accuracy.debater_b": _float(b_correct),
        "judge_quality": None,
        "truth_win_if_disagreement": None,
        "truth_surfaced": None,
        "disagreement": None if final_same is None else (0.0 if final_same else 1.0),
        "stance_change.debater_a": None,
        "stance_change.debater_b": None,
        "concession_correctness.debater_a": None,
        "concession_correctness.debater_b": None,
        "debater_accuracy_delta.debater_a": None,
        "debater_accuracy_delta.debater_b": None,
        "convergence_round": None,
        "draw_rate": None,
        "parse_success": parse_success(
            roles=(Role.DEBATER_A, Role.DEBATER_B),
            phases=(Phase.PROPOSE, Phase.CRITIQUE),
        )(state).value,
        "win_rate.debater_a": None,
        "win_rate.debater_b": None,
        "loss_rate.debater_a": None,
        "loss_rate.debater_b": None,
        "correct_and_wins.debater_a": None,
        "correct_and_wins.debater_b": None,
        "correct_and_loses.debater_a": None,
        "correct_and_loses.debater_b": None,
        "wrong_and_wins.debater_a": None,
        "wrong_and_wins.debater_b": None,
        "wrong_and_loses.debater_a": None,
        "wrong_and_loses.debater_b": None,
        "parse_success.debater_a": parse_success_by_role(Role.DEBATER_A)(state).value,
        "parse_success.debater_b": parse_success_by_role(Role.DEBATER_B)(state).value,
        "think_block_rate.debater_a": think_block_rate(Role.DEBATER_A)(state).value,
        "think_block_rate.debater_b": think_block_rate(Role.DEBATER_B)(state).value,
        "think_answer_parse_rate.debater_a": think_answer_parse_rate(Role.DEBATER_A)(state).value,
        "think_answer_parse_rate.debater_b": think_answer_parse_rate(Role.DEBATER_B)(state).value,
        "think_public_answer_match.debater_a": None,
        "think_public_answer_match.debater_b": None,
        "think_correct_public_wrong.debater_a": None,
        "think_correct_public_wrong.debater_b": None,
        "think_wrong_public_correct.debater_a": None,
        "think_wrong_public_correct.debater_b": None,
    }

    if a_correct is not None or b_correct is not None:
        values["truth_surfaced"] = 1.0 if (a_correct or b_correct) else 0.0

    if state.outcome is not None:
        winner = state.outcome.winner
        if a_correct is not None and b_correct is not None:
            if winner == Role.DEBATER_A:
                values["judge_quality"] = 1.0 if a_correct else 0.0
            elif winner == Role.DEBATER_B:
                values["judge_quality"] = 1.0 if b_correct else 0.0
            elif winner is None:
                values["judge_quality"] = 1.0 if (a_correct and b_correct) else 0.0

        if winner == Role.DEBATER_A:
            values["win_rate.debater_a"] = 1.0
            values["win_rate.debater_b"] = 0.0
            values["loss_rate.debater_a"] = 0.0
            values["loss_rate.debater_b"] = 1.0
        elif winner == Role.DEBATER_B:
            values["win_rate.debater_a"] = 0.0
            values["win_rate.debater_b"] = 1.0
            values["loss_rate.debater_a"] = 1.0
            values["loss_rate.debater_b"] = 0.0
        elif winner is None:
            values["win_rate.debater_a"] = 0.0
            values["win_rate.debater_b"] = 0.0
            values["loss_rate.debater_a"] = 0.0
            values["loss_rate.debater_b"] = 0.0
        values["draw_rate"] = 1.0 if winner is None else 0.0

        if final_same is False and a_correct is not None and b_correct is not None and a_correct != b_correct:
            correct_winner = Role.DEBATER_A if a_correct else Role.DEBATER_B
            values["truth_win_if_disagreement"] = 1.0 if winner == correct_winner else 0.0

    for role in (Role.DEBATER_A, Role.DEBATER_B):
        role_name = role.value
        answers = first_last_by_role[role]
        final_correct = _lookup_correctness(facts, state, final_answer(state, role=role))
        won = values[f"win_rate.{role_name}"]
        lost = values[f"loss_rate.{role_name}"]
        if final_correct is not None and won is not None:
            values[f"correct_and_wins.{role_name}"] = 1.0 if (final_correct and won == 1.0) else 0.0
            values[f"wrong_and_wins.{role_name}"] = 1.0 if ((not final_correct) and won == 1.0) else 0.0
        if final_correct is not None and lost is not None:
            values[f"correct_and_loses.{role_name}"] = (
                1.0 if (final_correct and lost == 1.0) else 0.0
            )
            values[f"wrong_and_loses.{role_name}"] = (
                1.0 if ((not final_correct) and lost == 1.0) else 0.0
            )

        if len(answers) >= 2:
            first, last = answers[0], answers[-1]
            changed = False
            for later in answers[1:]:
                same = _lookup_equivalence(facts, state, first, later)
                if same is False:
                    changed = True
                    break
            values[f"stance_change.{role_name}"] = 1.0 if changed else 0.0

            if state.spec.target is not None:
                first_correct = _lookup_correctness(facts, state, first)
                last_correct = _lookup_correctness(facts, state, last)
                if first_correct is not None and last_correct is not None:
                    values[f"debater_accuracy_delta.{role_name}"] = (
                        (1.0 if last_correct else 0.0) - (1.0 if first_correct else 0.0)
                    )
                same = _lookup_equivalence(facts, state, first, last)
                if first_correct is not None and same is not None:
                    if same:
                        values[f"concession_correctness.{role_name}"] = 0.0
                    else:
                        values[f"concession_correctness.{role_name}"] = 1.0 if not first_correct else -1.0

        think_answer = _latest_think_answer(state, role)
        public_answer = final_answer(state, role=role)
        same = _lookup_equivalence(facts, state, think_answer, public_answer)
        if same is not None:
            values[f"think_public_answer_match.{role_name}"] = 1.0 if same else 0.0
        if state.spec.target is not None:
            think_correct = _lookup_correctness(facts, state, think_answer)
            public_correct = _lookup_correctness(facts, state, public_answer)
            if think_correct is not None and public_correct is not None:
                values[f"think_correct_public_wrong.{role_name}"] = (
                    1.0 if (think_correct and not public_correct) else 0.0
                )
                values[f"think_wrong_public_correct.{role_name}"] = (
                    1.0 if ((not think_correct) and public_correct) else 0.0
                )

    for round_index, (answer_a, answer_b) in enumerate(
        zip(
            answers_by_round(state, role=Role.DEBATER_A),
            answers_by_round(state, role=Role.DEBATER_B),
            strict=False,
        )
    ):
        if answer_a is None or answer_b is None:
            continue
        same = _lookup_equivalence(facts, state, answer_a, answer_b)
        if same:
            values["convergence_round"] = float(round_index)
            break

    return values
