"""Evaluate semantic scorer prompt variants against a labeled adversarial bank."""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chz
import yaml
from jinja2 import Environment, StrictUndefined

from tinker_cookbook.scoring import BinaryJudgeBuilder
from tinker_cookbook.scoring.types import normalize_binary_verdict_token

TaskKind = Literal["grader", "matcher"]

_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)


@dataclass(frozen=True)
class PromptBlock:
    system: str
    user: str
    positive: str
    negative: str


@dataclass(frozen=True)
class PromptVariant:
    name: str
    matcher: PromptBlock
    grader: PromptBlock


@dataclass(frozen=True)
class BankCase:
    id: str
    task: TaskKind
    distribution: str
    question: str
    target: str | None
    response: str | None
    a: str | None
    b: str | None
    label: bool
    notes: str


@dataclass(frozen=True)
class CaseResult:
    variant: str
    case_id: str
    task: TaskKind
    distribution: str
    label: bool
    prediction: bool
    verdict: str | None
    raw_response: str
    valid_verdict: bool


@chz.chz
class Config:
    bank_path: str
    variants_path: str
    scorer_builder: BinaryJudgeBuilder
    parallelism: int | None = None
    output_path: str | None = None


def _validate_prompt_block(name: str, raw: object) -> PromptBlock:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping.")
    system = raw.get("system")
    user = raw.get("user")
    positive = raw.get("positive")
    negative = raw.get("negative")
    if not all(isinstance(value, str) for value in (system, user, positive, negative)):
        raise TypeError(f"{name} must define string system/user/positive/negative fields.")
    norm_positive = normalize_binary_verdict_token(positive)
    norm_negative = normalize_binary_verdict_token(negative)
    if norm_positive is None or norm_negative is None:
        raise ValueError(f"{name} verdict words must not be empty.")
    if norm_positive == norm_negative:
        raise ValueError(f"{name} positive/negative verdicts must differ after normalization.")
    return PromptBlock(
        system=system,
        user=user,
        positive=norm_positive,
        negative=norm_negative,
    )


def load_prompt_variants(path: str) -> list[PromptVariant]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("variants"), dict):
        raise ValueError("variants YAML must contain a top-level `variants:` mapping.")
    variants: list[PromptVariant] = []
    for name, body in raw["variants"].items():
        if not isinstance(name, str) or not isinstance(body, dict):
            raise ValueError("Each prompt variant must be a named mapping.")
        variants.append(
            PromptVariant(
                name=name,
                matcher=_validate_prompt_block(f"{name}.matcher", body.get("matcher")),
                grader=_validate_prompt_block(f"{name}.grader", body.get("grader")),
            )
        )
    if not variants:
        raise ValueError("No prompt variants found.")
    return variants


def _normalize_case_label(raw_label: object, *, source: str) -> bool:
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "positive", "same", "correct"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "negative", "different", "incorrect"}:
            return False
    raise ValueError(f"Unsupported label {raw_label!r} in {source}")


def _raw_case_to_bank_case(raw: dict[str, object], *, source: str) -> BankCase:
    task = raw["task"]
    if task not in ("grader", "matcher"):
        raise ValueError(f"Unsupported task {task!r} in {source}")
    target = None if raw.get("target") is None else str(raw["target"])
    response = None if raw.get("response") is None else str(raw["response"])
    a = None if raw.get("a") is None else str(raw["a"])
    b = None if raw.get("b") is None else str(raw["b"])

    if task == "matcher" and (a is None or b is None):
        a = a or target
        b = b or response
    if task == "grader" and (target is None or response is None):
        target = target or a
        response = response or b

    return BankCase(
        id=str(raw["id"]),
        task=task,
        distribution=str(raw["distribution"]),
        question=str(raw["question"]),
        target=target,
        response=response,
        a=a,
        b=b,
        label=_normalize_case_label(raw["label"], source=source),
        notes=str(raw["notes"]),
    )


def load_bank(path: str) -> list[BankCase]:
    cases: list[BankCase] = []
    bank_path = Path(path)
    if bank_path.is_dir():
        for child in sorted(bank_path.glob("*.json")):
            raw = json.loads(child.read_text())
            if not isinstance(raw, list):
                raise ValueError(f"Expected {child} to contain a JSON array.")
            for idx, case in enumerate(raw, start=1):
                if not isinstance(case, dict):
                    raise ValueError(f"Expected object at {child}[{idx}].")
                cases.append(_raw_case_to_bank_case(case, source=f"{child}[{idx}]"))
    else:
        with bank_path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                cases.append(_raw_case_to_bank_case(raw, source=f"{path}:{line_no}"))
    if not cases:
        raise ValueError(f"No bank cases found in {path}.")
    return cases


def _render_user(block: PromptBlock, case: BankCase) -> str:
    template = _JINJA_ENV.from_string(block.user)
    if case.task == "grader":
        if case.target is None or case.response is None:
            raise ValueError(f"grader case {case.id} is missing target/response")
        return template.render(
            question=case.question,
            target=case.target,
            response=case.response,
        )
    if case.a is None or case.b is None:
        raise ValueError(f"matcher case {case.id} is missing a/b")
    return template.render(
        question=case.question,
        a=case.a,
        b=case.b,
    )


async def _evaluate_case(
    semaphore: asyncio.Semaphore,
    scorer,
    variant: PromptVariant,
    case: BankCase,
) -> CaseResult:
    block = variant.grader if case.task == "grader" else variant.matcher
    user = _render_user(block, case)
    try:
        async with semaphore:
            raw_response = await scorer.complete(block.system, user)
    except Exception as exc:
        raw_response = f"[ERROR] {type(exc).__name__}: {exc}"
        return CaseResult(
            variant=variant.name,
            case_id=case.id,
            task=case.task,
            distribution=case.distribution,
            label=case.label,
            prediction=not case.label,
            verdict=None,
            raw_response=raw_response,
            valid_verdict=False,
        )
    verdict = normalize_binary_verdict_token(raw_response)
    if verdict == block.positive:
        prediction = True
        valid_verdict = True
    elif verdict == block.negative:
        prediction = False
        valid_verdict = True
    else:
        prediction = not case.label
        valid_verdict = False
    return CaseResult(
        variant=variant.name,
        case_id=case.id,
        task=case.task,
        distribution=case.distribution,
        label=case.label,
        prediction=prediction,
        verdict=verdict,
        raw_response=raw_response,
        valid_verdict=valid_verdict,
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2 * total)) / denom
    radius = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z * z) / (4 * total * total)))
        / denom
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _confusion(results: list[CaseResult]) -> dict[str, int]:
    tp = sum(1 for result in results if result.label and result.prediction)
    tn = sum(1 for result in results if (not result.label) and (not result.prediction))
    fp = sum(1 for result in results if (not result.label) and result.prediction)
    fn = sum(1 for result in results if result.label and (not result.prediction))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metric_bundle(results: list[CaseResult]) -> dict[str, float | int | list[float]]:
    confusion = _confusion(results)
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    total = len(results)
    positives = tp + fn
    negatives = tn + fp
    accuracy = (tp + tn) / total if total else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / positives if positives else math.nan
    specificity = tn / negatives if negatives else math.nan
    fpr = fp / negatives if negatives else math.nan
    fnr = fn / positives if positives else math.nan
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if not math.isnan(recall) and not math.isnan(specificity)
        else math.nan
    )
    acc_lo, acc_hi = _wilson_interval(tp + tn, total)
    invalid_verdicts = sum(1 for result in results if not result.valid_verdict)
    invalid_verdict_rate = invalid_verdicts / total if total else math.nan
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "accuracy_ci95": [acc_lo, acc_hi],
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": balanced_accuracy,
        "invalid_verdicts": invalid_verdicts,
        "invalid_verdict_rate": invalid_verdict_rate,
    }


def summarize_variant(results: list[CaseResult]) -> dict[str, object]:
    by_task: dict[str, list[CaseResult]] = defaultdict(list)
    by_distribution: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_task[result.task].append(result)
        by_distribution[result.distribution].append(result)

    errors = [
        {
            "case_id": result.case_id,
            "task": result.task,
            "distribution": result.distribution,
            "label": result.label,
            "prediction": result.prediction,
            "verdict": result.verdict,
            "raw_response": result.raw_response,
            "valid_verdict": result.valid_verdict,
        }
        for result in results
        if result.label != result.prediction or not result.valid_verdict
    ]

    return {
        "overall": _metric_bundle(results),
        "by_task": {name: _metric_bundle(group) for name, group in sorted(by_task.items())},
        "by_distribution": {
            name: _metric_bundle(group) for name, group in sorted(by_distribution.items())
        },
        "errors": errors,
    }


async def evaluate_variants(config: Config) -> dict[str, object]:
    bank = load_bank(config.bank_path)
    variants = load_prompt_variants(config.variants_path)
    scorer = config.scorer_builder.build()
    semaphore = asyncio.Semaphore(max(1, config.parallelism or config.scorer_builder.max_connections))

    results_by_variant: dict[str, list[CaseResult]] = {variant.name: [] for variant in variants}

    async def _run_variant(variant: PromptVariant) -> None:
        tasks = [
            _evaluate_case(semaphore, scorer, variant, case)
            for case in bank
        ]
        variant_results = await asyncio.gather(*tasks)
        results_by_variant[variant.name] = variant_results

    await asyncio.gather(*(_run_variant(variant) for variant in variants))

    summary = {
        "bank_path": config.bank_path,
        "variants_path": config.variants_path,
        "n_cases": len(bank),
        "variants": {
            variant_name: summarize_variant(results)
            for variant_name, results in sorted(results_by_variant.items())
        },
    }
    return summary


async def main(config: Config) -> None:
    summary = await evaluate_variants(config)
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
