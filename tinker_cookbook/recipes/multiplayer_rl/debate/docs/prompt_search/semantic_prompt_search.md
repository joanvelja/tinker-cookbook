# Semantic Scorer Prompt Search

## Setup

- Model: `gpt-5-mini` via the OpenAI-compatible adapter
- Bank: 100 adversarial cases across 10 distributions
- Tasks:
  - `grader`: response vs target correctness
  - `matcher`: answer-vs-answer equivalence
- Evaluation script:
  - [`eval_semantic_prompt_variants.py`](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/scripts/eval_semantic_prompt_variants.py)
- Candidate variants:
  - [`semantic_variants.yaml`](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/docs/prompt_search/semantic_variants.yaml)
- Visible backup prompt stash:
  - [`leading_backup_prompts.yaml`](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/docs/prompt_search/leading_backup_prompts.yaml)

## Bank Design

The bank mixes 10 targeted adversarial distributions:

- units / conversions / interval boundaries
- aliases / notation / date formats
- scope / subset / superset / completeness
- polarity / negation / ordering reversal
- temporal qualifiers / deadlines / versioning
- comparisons / weak vs strict inequalities
- sets / ordering invariance / duplicates
- causal paraphrase / cause-effect confusion
- coreference / deixis / ellipsis
- granularity / identifiers / label-vs-description

Each distribution contributes 5 grader cases and 5 matcher cases.

## Results Summary

### Wave 1

Best first-pass variant on the 100-case bank:

- `nanodebate_baseline`
  - overall accuracy: `0.94`
  - overall FPR: `0.08`
  - overall FNR: `0.04`
  - grader accuracy: `0.94`
  - matcher accuracy: `0.94`

The strongest non-baseline variants were more conservative, but they paid for that
with recall loss, especially on grader cases involving harmless context,
logical restatements, or equivalent representations.

### Repeatability Check

Repeated 3x on the two strongest candidates:

- `nanodebate_baseline`
  - mean overall accuracy: `0.930`
  - mean overall FPR: `0.0667`
  - mean overall FNR: `0.0733`
  - mean grader FPR: `0.0370`
  - mean grader FNR: `0.1159`
  - mean matcher FPR: `0.1014`
  - mean matcher FNR: `0.0370`

- `baseline_plus_disambiguation_v1`
  - mean overall accuracy: `0.9267`
  - mean overall FPR: `0.0400`
  - mean overall FNR: `0.1067`
  - mean grader FPR: `0.0000`
  - mean grader FNR: `0.1739`
  - mean matcher FPR: `0.0870`
  - mean matcher FNR: `0.0494`

Interpretation:

- The disambiguation-heavy variant did cut false positives.
- It also increased false negatives enough to lose on mean accuracy and on the
  symmetric error tradeoff used here.
- The baseline remained the best contained prompt overall.

## Chosen Prompt

Adopt the nanodebate baseline prompt family as the default scorer wording for now.

Reason:

- highest mean overall accuracy among tested contenders
- better recall than stricter variants
- simpler and shorter than the more engineered alternatives
- failures are narrow and diagnosable rather than broad and unstable

## Known Failure Modes

The chosen prompt still struggles on a small number of boundary cases:

- unresolved ambiguity in short forms
  - example: ambiguous abbreviations
  - example: locale-sensitive date formats
- entity granularity mismatch
  - city vs metropolitan area
- a few context-sensitive equivalence cases
  - broader context appended to the same entity
- some logically equivalent reformulations
  - negative-form restatements
- urgency / temporal nuance
  - `soon` vs `immediately`

These are the right targets for a future matcher-only follow-up search. The
current evidence does not justify replacing the baseline with a stricter
combined matcher+grader prompt.

## Fresh Matcher Holdout (2026-03-05)

- New bank:
  - [`matcher_holdout_2026-03-05.jsonl`](/Users/joalja/Documents/Github/ext/tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/debate/docs/prompt_search/matcher_holdout_2026-03-05.jsonl)
  - 50 matcher-only cases
  - 10 distributions, 5 cases each
  - balanced labels: 25 `SAME`, 25 `DIFFERENT`
- Goal:
  - test matcher-only prompt changes on a fresh unseen set before changing the shipped prompt

### Single-Pass Holdout Result

Best first pass on the fresh holdout:

- `scope_aware_qualifiers_v1`
  - accuracy: `0.96`
  - balanced accuracy: `0.96`
  - precision: `0.926`
  - recall: `1.00`
  - FPR: `0.08`
  - FNR: `0.00`

Tied on the same first-pass holdout metrics:

- `baseline_plus_question_scope_v1`

This was not enough to trust a prompt change. Single-pass swings were too large
in the earlier search.

### Repeat Study

Repeated 3x on both the old 100-case dev bank and the fresh 50-case holdout for
four serious contenders:

- `nanodebate_baseline`
- `compact_boundary`
- `baseline_plus_referent_granularity_v1`
- `baseline_plus_question_scope_v1`

Mean matcher metrics:

- Dev bank:
  - `nanodebate_baseline`: accuracy `0.940`, balanced accuracy `0.937`, FPR `0.101`, FNR `0.025`
  - `baseline_plus_referent_granularity_v1`: accuracy `0.933`, balanced accuracy `0.932`, FPR `0.087`, FNR `0.049`
  - `baseline_plus_question_scope_v1`: accuracy `0.927`, balanced accuracy `0.927`, FPR `0.072`, FNR `0.074`
  - `compact_boundary`: accuracy `0.920`, balanced accuracy `0.921`, FPR `0.072`, FNR `0.086`
- Fresh holdout:
  - `baseline_plus_referent_granularity_v1`: accuracy `0.940`, balanced accuracy `0.940`, FPR `0.120`, FNR `0.000`
  - `nanodebate_baseline`: accuracy `0.933`, balanced accuracy `0.933`, FPR `0.133`, FNR `0.000`
  - `baseline_plus_question_scope_v1`: accuracy `0.927`, balanced accuracy `0.927`, FPR `0.120`, FNR `0.027`
  - `compact_boundary`: accuracy `0.900`, balanced accuracy `0.900`, FPR `0.187`, FNR `0.013`

Interpretation:

- No contender dominates the baseline across both banks and repeated runs.
- `baseline_plus_referent_granularity_v1` is the strongest alternative:
  - slightly lower false-`SAME` rate than baseline
  - slight recall cost on the dev bank
  - exact total correctness tie with baseline across the repeated dev+holdout runs
- `baseline_plus_question_scope_v1` and `compact_boundary` both looked better in
  one slice than they did under repeats.

### Recommendation After Repeats

Keep the shipped matcher prompt unchanged for now.

Reason:

- the baseline remains the most recall-stable option
- the strongest alternative improves specificity only modestly
- the strongest alternative does not fix the most recurrent matcher failures
- the repeat study does not clear a high enough bar for a live prompt swap

If we decide to bias toward lower false `SAME` calls in the future, the best
contained candidate is `baseline_plus_referent_granularity_v1`. Right now it is
best treated as the leading backup, not the new default.
