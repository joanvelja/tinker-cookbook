"""Post-episode semantic fact resolution for debate metrics."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from ..prompts import BinaryJudgeTemplate, DebatePrompts, normalize_binary_verdict_token
from ..types import DebateState, Phase, Role, ScoringMode
from ..think import has_think_block
from .mcq import normalize_mcq
from .metrics import (
    _extract_thinking_text,
    _parse_think_answer_strict,
    parse_success,
)
from .providers import AnswerJudgeClient
from .trajectory import answer_from_utterance

logger = logging.getLogger(__name__)

MatcherKey = tuple[str, str, str, str]
GraderKey = tuple[str, str, str, str]

_DEBATER_ROLES = (Role.DEBATER_A, Role.DEBATER_B)


class BinaryJudgeError(ValueError):
    """Raised when a binary judge returns an invalid verdict."""


@dataclass(frozen=True)
class TranscriptSummary:
    """Pre-computed transcript statistics from a single forward pass."""

    final_answers: dict[Role, str | None]
    answers_by_round: dict[Role, list[str | None]]
    non_null_answers: dict[Role, list[str]]
    think_answers: dict[Role, str | None]
    parse_success_count: dict[Role, int]
    total_utterance_count: dict[Role, int]
    think_block_count: dict[Role, int]
    parseable_think_count: dict[Role, int]


def summarize_transcript(state: DebateState) -> TranscriptSummary:
    """Compute all transcript-derived statistics in one forward pass."""
    schedule = state.spec.schedule
    num_rounds = max((s.round_index for s in schedule), default=0) + 1

    answers_by_round: dict[Role, list[str | None]] = {r: [None] * num_rounds for r in _DEBATER_ROLES}
    parse_success_count: dict[Role, int] = {r: 0 for r in _DEBATER_ROLES}
    total_utterance_count: dict[Role, int] = {r: 0 for r in _DEBATER_ROLES}
    think_block_count: dict[Role, int] = {r: 0 for r in _DEBATER_ROLES}
    parseable_think_count: dict[Role, int] = {r: 0 for r in _DEBATER_ROLES}
    latest_think_answer: dict[Role, str | None] = {r: None for r in _DEBATER_ROLES}
    last_answer: dict[Role, str | None] = {r: None for r in _DEBATER_ROLES}

    for u in state.transcript:
        if u.role not in _DEBATER_ROLES:
            continue
        role = u.role
        total_utterance_count[role] += 1

        ans = answer_from_utterance(u)
        if ans is not None:
            parse_success_count[role] += 1
            last_answer[role] = ans
            if u.round_index < num_rounds:
                answers_by_round[role][u.round_index] = ans

        if has_think_block(u.text):
            think_block_count[role] += 1
            reasoning = _extract_thinking_text(u.text)
            if reasoning is not None:
                parsed = _parse_think_answer_strict(reasoning)
                if parsed is not None:
                    parseable_think_count[role] += 1
                    latest_think_answer[role] = parsed

    non_null_answers: dict[Role, list[str]] = {}
    for role in _DEBATER_ROLES:
        rounds = answers_by_round[role]
        non_null_answers[role] = [a for a in rounds if a is not None]

    return TranscriptSummary(
        final_answers=last_answer,
        answers_by_round=answers_by_round,
        non_null_answers=non_null_answers,
        think_answers=latest_think_answer,
        parse_success_count=parse_success_count,
        total_utterance_count=total_utterance_count,
        think_block_count=think_block_count,
        parseable_think_count=parseable_think_count,
    )


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


_NEGATION_CUES = re.compile(r"\b(?:not|isn't|neither|no)\b", re.IGNORECASE)


def _tolerant_extract(response_text: str, positive: str, negative: str) -> str | None:
    """Fallback extraction: scan for standalone canonical tokens with negation awareness.

    Returns the canonical token (positive/negative) ONLY IF exactly one is found
    with zero negation cues within ~3 preceding tokens. Otherwise returns None.
    """
    tokens = response_text.split()
    counts: dict[str, int] = {positive: 0, negative: 0}
    negated = False

    for i, token in enumerate(tokens):
        normalized = token.strip().upper().rstrip(".,!?;:'\"")
        if normalized not in (positive, negative):
            continue
        # Check for negation cues within ~3 preceding tokens.
        window_start = max(0, i - 3)
        preceding = " ".join(tokens[window_start:i])
        if _NEGATION_CUES.search(preceding):
            negated = True
            break
        counts[normalized] += 1

    if negated:
        return None

    found = [canon for canon, n in counts.items() if n > 0]
    if len(found) == 1 and counts[found[0]] >= 1:
        return found[0]
    return None


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

    # Tolerant fallback: word-boundary scan with negation check.
    tolerant = _tolerant_extract(response_text, template.positive, template.negative)
    if tolerant == template.positive:
        return True
    if tolerant == template.negative:
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


async def resolve_debate_facts_for_states(
    states: list[DebateState],
    *,
    scorer: AnswerJudgeClient | None,
    prompts_for_ref: Callable[[str], DebatePrompts],
    parallelism: int,
    strict: bool = True,
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
        key = _matcher_key(state.spec.problem.scoring_mode, question, left, right)
        if key in equivalence_results or key in equivalence_tasks:
            telemetry["cache_hits"] += 1
            return key

        telemetry["cache_misses"] += 1
        if state.spec.problem.scoring_mode == ScoringMode.MCQ:
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
        key = _grader_key(state.spec.problem.scoring_mode, question, target, response)
        if key in correctness_results or key in correctness_tasks:
            telemetry["cache_hits"] += 1
            return key

        telemetry["cache_misses"] += 1
        if state.spec.problem.scoring_mode == ScoringMode.MCQ:
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
        summary = summarize_transcript(state)
        question = state.spec.problem.task_prompt
        prompts = _prompts(state.spec.prompts_ref) if state.spec.problem.scoring_mode == ScoringMode.OPEN_ENDED else None
        equivalence_keys: set[MatcherKey] = set()
        correctness_keys: set[GraderKey] = set()

        final_a = summary.final_answers[Role.DEBATER_A]
        final_b = summary.final_answers[Role.DEBATER_B]
        if final_a is not None and final_b is not None:
            equivalence_keys.add(
                _schedule_equivalence(state, final_a, final_b, question=question, prompts=prompts)
            )

        for role in _DEBATER_ROLES:
            answers = summary.non_null_answers[role]
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

        rounds_a = summary.answers_by_round[Role.DEBATER_A]
        rounds_b = summary.answers_by_round[Role.DEBATER_B]
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

        if state.spec.problem.target is not None:
            target = state.spec.problem.target
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
            for role in _DEBATER_ROLES:
                for answer in summary.non_null_answers[role]:
                    correctness_keys.add(
                        _schedule_correctness(
                            state,
                            answer,
                            target,
                            question=question,
                            prompts=prompts,
                        )
                    )
                think_answer = summary.think_answers[role]
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

        for role in _DEBATER_ROLES:
            think_answer = summary.think_answers[role]
            public_answer = summary.final_answers[role]
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
                scoring_mode=state.spec.problem.scoring_mode,
                equivalence_keys=frozenset(equivalence_keys),
                correctness_keys=frozenset(correctness_keys),
            )
        )

    async def _collect(
        tasks: dict[tuple[str, ...], asyncio.Task[bool]],
        sink: dict[tuple[str, ...], bool],
        *,
        kind: str,
    ) -> int:
        """Gather task results into sink. Returns count of BinaryJudgeErrors skipped."""
        if not tasks:
            return 0
        if strict:
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
            return 0

        # Non-strict: isolate BinaryJudgeErrors, re-raise everything else.
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        skipped = 0
        for key, result in zip(tasks, results, strict=True):
            if isinstance(result, BinaryJudgeError):
                logger.warning(
                    "Skipping %s key %s: %s", kind, key, result,
                )
                skipped += 1
            elif isinstance(result, BaseException):
                # Infrastructure failure — cancel remaining and re-raise.
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
                raise result
            else:
                sink[key] = result
        return skipped

    try:
        matcher_errors = await _collect(
            equivalence_tasks, equivalence_results, kind="matcher",
        )
        grader_errors = await _collect(
            correctness_tasks, correctness_results, kind="grader",
        )
    except Exception:
        outstanding = [*equivalence_tasks.values(), *correctness_tasks.values()]
        for task in outstanding:
            if not task.done():
                task.cancel()
        if outstanding:
            await asyncio.gather(*outstanding, return_exceptions=True)
        raise

    if not strict:
        telemetry["binary_judge_errors.matcher"] = matcher_errors
        telemetry["binary_judge_errors.grader"] = grader_errors

    resolved: list[ResolvedDebateFacts] = []
    for plan in plans:
        eq_dict = {
            key: equivalence_results[key]
            for key in plan.equivalence_keys
            if key in equivalence_results
        }
        corr_dict = {
            key: correctness_results[key]
            for key in plan.correctness_keys
            if key in correctness_results
        }

        plan_telemetry = dict(telemetry)
        if not strict:
            plan_telemetry["missing_equivalence_keys"] = (
                len(plan.equivalence_keys) - len(eq_dict)
            )
            plan_telemetry["missing_correctness_keys"] = (
                len(plan.correctness_keys) - len(corr_dict)
            )

        resolved.append(
            ResolvedDebateFacts(
                scoring_mode=plan.scoring_mode,
                equivalence=_freeze_mapping(eq_dict),
                correctness=_freeze_mapping(corr_dict),
                telemetry=_freeze_mapping(plan_telemetry),
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
    key = _matcher_key(facts.scoring_mode, state.spec.problem.task_prompt, left, right)
    return facts.equivalence.get(key)


def _lookup_correctness(
    facts: ResolvedDebateFacts,
    state: DebateState,
    response: str | None,
) -> bool | None:
    if response is None or state.spec.problem.target is None:
        return None
    key = _grader_key(facts.scoring_mode, state.spec.problem.task_prompt, state.spec.problem.target, response)
    return facts.correctness.get(key)


def _float(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return numerator/denominator, or None if denominator is 0."""
    return numerator / denominator if denominator > 0 else None


def built_in_metric_values(
    state: DebateState,
    facts: ResolvedDebateFacts,
    summary: TranscriptSummary | None = None,
) -> dict[str, float | None]:
    if summary is None:
        summary = summarize_transcript(state)

    final_a = summary.final_answers[Role.DEBATER_A]
    final_b = summary.final_answers[Role.DEBATER_B]

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
        # Phase-filtered parse_success kept as direct call (needs PROPOSE+CRITIQUE filter)
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
        "parse_success.debater_a": _safe_ratio(
            summary.parse_success_count[Role.DEBATER_A],
            summary.total_utterance_count[Role.DEBATER_A],
        ),
        "parse_success.debater_b": _safe_ratio(
            summary.parse_success_count[Role.DEBATER_B],
            summary.total_utterance_count[Role.DEBATER_B],
        ),
        "think_block_rate.debater_a": _safe_ratio(
            summary.think_block_count[Role.DEBATER_A],
            summary.total_utterance_count[Role.DEBATER_A],
        ),
        "think_block_rate.debater_b": _safe_ratio(
            summary.think_block_count[Role.DEBATER_B],
            summary.total_utterance_count[Role.DEBATER_B],
        ),
        "think_answer_parse_rate.debater_a": _safe_ratio(
            summary.parseable_think_count[Role.DEBATER_A],
            summary.think_block_count[Role.DEBATER_A],
        ),
        "think_answer_parse_rate.debater_b": _safe_ratio(
            summary.parseable_think_count[Role.DEBATER_B],
            summary.think_block_count[Role.DEBATER_B],
        ),
        "think_public_answer_match.debater_a": None,
        "think_public_answer_match.debater_b": None,
        "think_correct_public_wrong.debater_a": None,
        "think_correct_public_wrong.debater_b": None,
        "think_wrong_public_correct.debater_a": None,
        "think_wrong_public_correct.debater_b": None,
    }

    if a_correct is True or b_correct is True:
        values["truth_surfaced"] = 1.0
    elif a_correct is False and b_correct is False:
        values["truth_surfaced"] = 0.0
    # else: stays None (one or both unknown)

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

    for role in _DEBATER_ROLES:
        role_name = role.value
        answers = summary.non_null_answers[role]
        final_correct = _lookup_correctness(facts, state, summary.final_answers[role])
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
            has_unknown = False
            for later in answers[1:]:
                same = _lookup_equivalence(facts, state, first, later)
                if same is False:
                    changed = True
                    break
                if same is None:
                    has_unknown = True
            if changed:
                values[f"stance_change.{role_name}"] = 1.0
            elif has_unknown:
                values[f"stance_change.{role_name}"] = None
            else:
                values[f"stance_change.{role_name}"] = 0.0

            if state.spec.problem.target is not None:
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

        think_answer = summary.think_answers[role]
        public_answer = summary.final_answers[role]
        same = _lookup_equivalence(facts, state, think_answer, public_answer)
        if same is not None:
            values[f"think_public_answer_match.{role_name}"] = 1.0 if same else 0.0
        if state.spec.problem.target is not None:
            think_correct = _lookup_correctness(facts, state, think_answer)
            public_correct = _lookup_correctness(facts, state, public_answer)
            if think_correct is not None and public_correct is not None:
                values[f"think_correct_public_wrong.{role_name}"] = (
                    1.0 if (think_correct and not public_correct) else 0.0
                )
                values[f"think_wrong_public_correct.{role_name}"] = (
                    1.0 if ((not think_correct) and public_correct) else 0.0
                )

    rounds_a = summary.answers_by_round[Role.DEBATER_A]
    rounds_b = summary.answers_by_round[Role.DEBATER_B]
    prior_all_resolved = True  # All earlier non-None rounds resolved to False
    for round_index, (answer_a, answer_b) in enumerate(
        zip(rounds_a, rounds_b, strict=False)
    ):
        if answer_a is None or answer_b is None:
            continue
        same = _lookup_equivalence(facts, state, answer_a, answer_b)
        if same is True and prior_all_resolved:
            values["convergence_round"] = float(round_index)
            break
        if same is None:
            prior_all_resolved = False
        # same is False: prior_all_resolved stays as-is (correct: disagreement before)

    return values
