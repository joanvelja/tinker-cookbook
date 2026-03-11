# Gate Review Log — Scoring Pipeline

Tracking all gate findings, corrections, and open items.

## Gate-0 (scoring.py, parsing.py, mcq.py)

### Fixed
- `scoring.py:16` — comment referenced "nanodebate" → removed

### Open / Deferred
- **Bool coercion accepts invalid values as False** (parsing.py:29) — everything not in ("true","yes","1") silently maps to False. Should return None for unrecognized values. [Codex, MEDIUM]
- **Duplicate XML tags: silent last-write-wins** (parsing.py:58) — repeated tags overwrite earlier values. Adversarial/malformed outputs can steer extraction. [Codex, MEDIUM]
- **`_validate` brittle on malformed configs** — non-dict YAML root raises AttributeError instead of clean ValueError. Non-string binary values crash on .strip(). Non-numeric bounds crash in math.isfinite. [Codex, MEDIUM]
- **Latent import cycle risk** — scoring→mcq→parsing→scoring chain is safe now (lazy imports) but any refactor moving to module level would create hard cycle. Consider extracting shared regex to _common.py if this becomes a problem. [Codex, MEDIUM]
- **MCQ false positives on "A or B"** without "both"/"either" prefix — `_extract_terminal` returns B. Inherited from nanodebate. [Codex, MEDIUM]
- **Nested/malformed XML tags produce partial values** — regex is non-greedy, `<a><b></a></b>` gives partial. Inherited from nanodebate. [Codex, MEDIUM]
- **Stale comments** referencing format.py, resolve.py in a few places after merge. [Codex, LOW]

## Gate-1 (types.py, prompts/__init__.py, scientific_mcq.yaml)

### Fixed
- **Judge tag mismatch**: scientific_mcq.yaml used `reasoning`, judge.py expects `reason` → renamed to `reason`
- **get_field_specs returned mutable internal dict** from LRU-cached object → now returns defensive copy
- **generate_format_instructions inconsistency**: preamble said "XML tags" but scored fields got bullet-point hints → fixed to use XML format throughout

### Fixed (post-gate)
- **check_ab_symmetry now covers `fields` section** — compares trigger sets and field-name sets per trigger between debater_a/b. Tests for asymmetric triggers and asymmetric field names added.

### Open / Deferred
- **Private API coupling**: prompts/__init__.py imports `_resolve_fields` and `_TYPE_MAP` from scoring.py. Internal refactors could break prompt loading. [Codex, LOW]

## Gate-2 (reducer.py, runtime.py, trajectory.py, metrics.py)

### Fixed
- (none — escalated item resolved, see below)

### RESOLVED
- **concession_correctness wrong→different_wrong = +1.0** — User confirmed: +1.0 is correct. Measures willingness to revise when wrong; debater_accuracy_delta separately tracks whether revision improved accuracy. Conflating both into one metric makes training dynamics harder to diagnose.

### Open / Deferred
- ~~**runtime.py judge extraction uses slot phase not trigger**~~ — **FIXED**: added `_PHASE_TO_TRIGGER` mapping in runtime.py: `judge_verdict→final`, `judge_query→boundary`. Debater phases pass through unchanged.
- **Trajectory backfill on parse failure** — final_answer/answer_at_round scan backward past fields=None utterances to find last known answer. By design (parse failure shouldn't erase stance; parse_success metric captures extraction failures separately). But can hide extraction issues from accuracy metrics. [Codex, HIGH, by-design]
- **Inconsistent winner=Role.JUDGE handling** — judge_quality returns N/A, truth_win returns 0.0. Practically irrelevant (JUDGE can't be winner), but inconsistent. [Codex, MEDIUM]
- **judge_quality with no transcript + existing outcome** — empty transcript means a_correct=False, b_correct=False → emits 0.0/1.0 rather than N/A. Edge case. [Codex, MEDIUM]
- **answers_by_round empty schedule** — returns [None] not []. max(..., default=0)+1 = 1. Harmless. [Codex, LOW]
- **Shallow freeze on Utterance.fields** — currently fine (field values are str/int/float scalars). If list-valued fields appear, need deep freeze. [Codex Gate-0 finding, carried forward]

### Test coverage gaps noted by Codex
- No test for winner=Role.JUDGE branch
- No test for critique overriding propose in same round
- No test for wrong→different-wrong in concession_correctness
- No test for stale answer backfill (latest utterance with fields=None)
- No test for empty schedule edge case
- No runtime integration test for judge field extraction

## Gate-3 (judge.py, env.py, test_integration.py)

### Fixed
- **Judge partial extraction forces tie** — if schema extraction returns `{'reason': '...'}` but no `decision`, regex fallback didn't fire → added fallback merge: `if "decision" not in fields: fallback = _extract_xml_fields(text); fallback.update(fields); fields = fallback` [Codex, HIGH]
- **Hardcoded absolute path in test** — `test_integration.py` had `cwd="/Users/joalja/..."` → replaced with `cwd=str(Path(__file__).resolve().parents[4])` [Codex, MEDIUM]

### Open / Deferred
- ~~**Frozen-opponent deadlock with include_judge_turns**~~ — **FIXED**: frozen-opponent `make_envs()` now validates schedule actors ⊆ {DEBATER_A, DEBATER_B}. Raises ValueError mentioning `include_judge_turns` if JUDGE is in the schedule.
- **Shared metrics dict in compute_group_rewards** — all problems in a batch share the same `mcq_debate_metrics()` dict reference. Harmless (MetricFn is pure), but could surprise if anyone mutates the dict. [Codex, LOW]

### Test coverage (27 integration tests added)
- Field extraction roundtrip (XML → extract_fields → Utterance.fields → trajectory → metric)
- Full 2-round debate metrics (accuracy, judge_quality, truth_surfaced, stance_change, concession_correctness, disagreement, debater_accuracy_delta)
- Disagreement + truth_win under different judge picks
- No-target and no-outcome edge cases (all metrics return None)
- Judge verdict parsing (schema, regex fallback, tie, garbage)
- dump_io with scientific_mcq prompts
