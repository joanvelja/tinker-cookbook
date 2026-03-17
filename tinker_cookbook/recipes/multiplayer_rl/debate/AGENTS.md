# Debate Recipe — Agent Instructions

Workflow norms for the debate recipe. Project-specific docs are in `README.md`.

## Greenfield Mode

NO REGARDS FOR BACKWARDS COMPATIBILITY UNLESS WE ARE NOT IN GREENFIELD DEV MODE. Always double check with user at the beginning of session.

## Development Environment

```bash
# Python 3.12+ required
uv sync                    # install deps (ALWAYS use uv, never pip)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests
```

Running any python command like `python -m pytest` **must** be done via `uv` to ensure the correct environment is activated, e.g. `uv run pytest`, `uv run python ...`, `uv run python -m pytest ...`, etc.

## Bug Fixes: Prove It Pattern

When given a bug or error report, the first step is to spawn a subagent to write a test that reproduces the issue. Only proceed once reproduction is confirmed.

Test level hierarchy — Reproduce at the lowest level that can capture the bug:

1. **Unit test** — Pure logic bugs, isolated functions (lives next to the code)
2. **Integration test** — Component interactions, API boundaries (lives next to the code)
3. **UX spec test** — Full user flows, UI-dependent behavior

For every bug fix:

1. **Reproduce with subagent** — Spawn a subagent to write a test that demonstrates the bug. The test should *fail* before the fix.
2. **Fix** — Implement the fix.
3. **Confirm** — The test now *passes*, proving the fix works.

If the bug is truly environment-specific or transient, document why a test isn't feasible rather than skipping silently.

## Hard Cutover Policy

Hard cutover by milestone; no multi-milestone dual runtime APIs. No class/registry/builder frameworks for prompt behavior. When migrating, you migrate or you don't — no shims, no dual paths.

## Plan Mode

- Concise plans (sacrifice grammar for brevity)
- End plans with unresolved questions
- Extensive use of Tasks (for state tracking), Agents Team (for parallel workflows), and Swarms (for exploration purposes).

## Spot-Check Tool

Monitor training run health:

```bash
# Single run
uv run python -m scripts.spot_check logs/thinking-experiment/rung1-no-think

# Compare runs
uv run python -m scripts.spot_check logs/thinking-experiment/rung*

# Live monitoring
uv run python -m scripts.spot_check logs/thinking-experiment/rung* --watch

# Full detail (secondary metrics + auto-discovered keys)
uv run python -m scripts.spot_check logs/thinking-experiment/rung1-no-think --verbose
```

5 gates: **SIGNAL** (learning signal alive?), **JUDGE** (judge trustworthy?), **SYMMETRY** (protocol fair?), **QUALITY** (pipeline healthy?), **BUDGET** (spending wisely?). Each shows OK/WARN/FAIL with sparkline trends.

Episode-derived signals (sycophancy, bullshit contest, agree-B-win) require `step`+`split` fields in episodes.jsonl (schema v3).

## Training Launch Notes

- Use `num_minibatches=8` for ~1.19x throughput (overlaps sampling with training, no off-policy risk). Derived from BOTEC on actual timing data: `step_time(M) = E + S + R + P/M`.
- Async RL is supported by the generic framework but not wired for debate. Streaming minibatch captures ~97% of async's benefit without off-policy complexity.

## Philosophy

This codebase will outlive you. Every shortcut, hacky implementation, non-extensible script, or ad-hoc solution becomes someone else's burden. Every hack compounds into technical debt that slows the whole team down. You are not just writing code. You are shaping the future of this project. The patterns you establish will be copied. The corners you cut will be cut again.

Fight entropy. Leave the codebase better than you found it: for each proposed change, examine the existing system and redesign it into the most elegant solution that would have emerged if the change had been a foundational assumption from the start.
