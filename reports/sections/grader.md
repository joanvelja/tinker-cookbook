# Grader Analysis: False Negatives and the Composite Grader

## The Problem

The LLM grader (gpt-5-mini with math-aware prompt) is the sole source of correctness reward. False negatives (correct answers graded as incorrect) create a toxic training signal: the model learns to avoid answer formats that confuse the grader rather than learning to solve problems correctly. False positives (wrong answers graded as correct) are less damaging because they dilute the reward signal without creating active misalignment.

## False Negative Rates by Model

From manual inspection of wrong-but-boxed responses across all three v3b runs:

| Model | FN rate | Sample size | Primary failure modes |
|-------|---------|-------------|----------------------|
| GPT-OSS v3b | 14% | 1/7 inspected | Symbolic notation vs English prose |
| Qwen3 Instruct v3b | 24% | 5/21 inspected | LaTeX wrappers, variable names, double-boxing |
| Qwen3 Think v3b | bug found | | `\boxed{answer}` literal accepted as correct |

The Qwen Instruct FN rate of 24% is alarmingly high. One in four correct-but-boxed answers receives zero reward during training. Over 50 training steps with 2,048 episodes each, this represents thousands of incorrectly penalized correct responses.

## Specific Failure Modes

### LaTeX Wrapper Confusion
Model outputs `\text{D}`, reference is `D`. The grader treats the LaTeX `\text{}` wrapper as semantically meaningful and grades INCORRECT. `\text{D}` renders as "D"; this is purely a notation difference.

### Variable Name Substitution
Model outputs `f(n) = n`, reference is `f(x) = x`. The grader compares string representations and flags the different variable name. Mathematically identical functions.

### Stochastic Grading Error
The grader occasionally rejects simple correct answers (`1` vs `1`) with no apparent reason. The LLM judge produces different verdicts on identical inputs across runs. Measured reproducibility rate: ~96% (4% of gradings flip between runs on the same input).

### Double-Boxing
Model writes `\boxed{7 \times 2023}` in the reasoning then `\boxed{C}` at the end. The `extract_boxed` function takes the last `\boxed{}` occurrence, so the grader receives "C" instead of "7 x 2023". This is a parsing bug that manifests as a grader false negative.

### The Biconditional Case Study
One initially suspected false negative turned out to be genuine: the model used the biconditional symbol (iff, $\iff$) where the correct answer required conjunction (AND). Verified with gpt-5 as a second opinion. The model's answer is mathematically distinct. A subtle error that looks like a notation difference but changes the truth conditions. This case validates that not all suspected FNs are real FNs.

## Impact on Training Dynamics

The grader FN signal is visible in the Qwen Instruct failure mode taxonomy from transcript probes:

| Category | Step 0 | Step 15 | Delta |
|----------|--------|---------|-------|
| Grader FN (correct but scored wrong) | 4/32 | 1/32 | -3 |
| Wrong approach (genuinely wrong) | 2/32 | 4/32 | +2 |

The model reduced grader FNs by 3 over 15 steps. It learned to avoid answer formats that confuse the grader. This came at the cost of 2 additional genuinely wrong answers. The model shifted from "correct in unusual format" to "wrong in standard format." True accuracy (including grader FN corrections) declined slightly: 18/32 to 17/32.

This is the textbook failure mode of reward misalignment: the model optimizes the proxy (grader score) at the expense of the target (mathematical correctness).

## The Composite Grader

To address grader false negatives, we built a composite grader with two stages.

Stage 1: SymPy (deterministic, <1ms). Parse both answers with SymPy and check mathematical equivalence. If SymPy says EQUAL, trust it. SymPy's True is reliable because it proves exact symbolic equivalence.

Stage 2: LLM fallback (stochastic, ~0.5-1s). If SymPy says False or errors out, fall through to the LLM grader with an improved prompt including negative examples. SymPy's False can miss semantic equivalences (prose descriptions, unconventional notation, function definitions). The LLM catches what SymPy misses.

```python
class CompositeGrader:
    async def grade(self, question, reference, extracted):
        sympy_result = self._sympy_grade(extracted, reference)
        if sympy_result is True:
            return GradeResult(correct=True, status="correct", detail="sympy")
        return await self._llm_grade(question, reference, extracted)
```

## Stress Test Results

The composite grader was tested against the LLM-only grader on real transcript data from all three v3b runs. The stress test script (`scripts/stress_test_composite_vs_llm_real.py`) extracts wrong-but-boxed response pairs from training logs and re-grades them with both graders.

Test set: up to 150 pairs per run, 310 total pairs. Source: wrong-but-boxed responses with known reference answers from correct trajectories in the same group.

| Metric | Count |
|--------|-------|
| Total pairs tested | 310 |
| Both agree INCORRECT | 261 |
| Both agree CORRECT | 22 |
| FN fixed (INCORRECT to CORRECT) | 27 |
| FP introduced (CORRECT to INCORRECT) | 0 |

The composite grader fixed 27 false negatives (8.7% of tested pairs) without introducing any false positives. Of the 27 fixes, some came through the SymPy fast path (symbolically equivalent answers with different notation or reordered terms) and others through the improved LLM prompt (LaTeX wrappers, variable names).

Zero false positives is the critical safety property. The composite grader is strictly better than the LLM-only grader on this test set.

## Grader Speed

SymPy grading takes <1ms per pair. For pairs where SymPy returns True, the composite grader avoids the LLM API call entirely, reducing both latency and cost. At 2,048 episodes per training step, even a modest SymPy hit rate saves meaningful API spend. The grading phase is small relative to sampling (embedded in the sampling wall time), so the larger benefit is improved determinism: SymPy-resolved grades are reproducible across runs, eliminating the 4% stochastic flip rate of LLM-only grading.

## Updated Grader Prompt

The v3b training runs used the original LLM grader prompt. The composite grader's LLM fallback uses an improved prompt with explicit examples covering the observed failure modes:

Original:
```
CORRECT = mathematically equivalent (notation/format differences are fine).
```

Updated (includes examples):
```
CORRECT: '\frac{1}{2}' vs '0.5' — same value.
CORRECT: '3\pi' vs '3π' — same expression.
INCORRECT: '\frac{1}{2}' vs '\frac{1}{3}' — different fraction.
INCORRECT: 'x^2 + 1' vs 'x^2 - 1' — different sign.
```

The updated prompt was committed for future runs but does not affect the v3b results reported here.

## Recommendations

Switch to composite grader for all future runs. Zero FP risk, 8.7% FN improvement. No downside.

Fix the extract_boxed double-boxing bug. When the model produces multiple `\boxed{}` occurrences, the parser should take the last one in the final channel (not analysis channel) or flag ambiguous cases for review.

Monitor grader FN rate as a training diagnostic. A declining FN rate during training may indicate the model is learning to avoid correct-but-awkward formats, which is a proxy optimization signal worth investigating.

Consider deterministic grading (SymPy-only) for problems with constrained answer formats (numeric, simple algebraic). Reserve LLM grading for open-ended answer formats where symbolic comparison is insufficient.
