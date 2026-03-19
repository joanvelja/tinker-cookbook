"""
Gradient budget allocation analysis from REAL GPQA RLVR training data.

Parses logtree HTML trace files, extracts episodes, computes advantages
using the real _normalize_subgroup function, and reports gradient budget
allocation across categories.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from bs4 import BeautifulSoup

# Import the real advantage function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinker_cookbook.rl.data_processing import _normalize_subgroup


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_trace_file(path: Path) -> list[dict]:
    """Parse a single logtree HTML trace file, return list of episode dicts."""
    html = path.read_text()
    soup = BeautifulSoup(html, "html.parser")
    ps = soup.find_all("p", class_="lt-p")

    episodes = []
    i = 0
    while i < len(ps):
        text = ps[i].get_text().strip()
        if text.startswith("Problem:"):
            problem_text = text
            # Next: Response
            if i + 1 < len(ps):
                response_text = ps[i + 1].get_text().strip()
            else:
                i += 1
                continue
            # Next: Reference Answer
            if i + 2 < len(ps):
                ref_text = ps[i + 2].get_text().strip()
            else:
                i += 1
                continue
            # Next: Scoring line
            if i + 3 < len(ps):
                score_text = ps[i + 3].get_text().strip()
            else:
                i += 1
                continue

            if not score_text.startswith("Boxed:"):
                i += 1
                continue

            # Parse scoring line: "Boxed: ✓, EOS: ✓, Correct: ✓, Reward: 1.00"
            boxed = "✓" in score_text.split("Boxed:")[1].split(",")[0]
            eos = "✓" in score_text.split("EOS:")[1].split(",")[0]
            correct = "✓" in score_text.split("Correct:")[1].split(",")[0]
            reward_match = re.search(r"Reward:\s*([-\d.]+)", score_text)
            reward = float(reward_match.group(1)) if reward_match else 0.0

            # Response length
            resp = response_text
            if resp.startswith("Response:"):
                resp = resp[len("Response:"):].strip()
            resp_chars = len(resp)
            resp_words = len(resp.split())
            resp_tokens_est = resp_chars / 4  # rough proxy

            # Category
            if not eos:
                category = "truncated"
            elif not correct:
                category = "wrong"
            else:
                category = "correct"

            episodes.append({
                "problem": problem_text[:200],  # truncate for grouping
                "response_chars": resp_chars,
                "response_words": resp_words,
                "response_tokens_est": resp_tokens_est,
                "reward": reward,
                "boxed": boxed,
                "eos": eos,
                "correct": correct,
                "category": category,
            })
            i += 4
        else:
            i += 1

    return episodes


def parse_run(run_dir: Path) -> dict[int, list[dict]]:
    """Parse all train_iteration_*.html files in a run directory.
    Returns {step: [episodes]}."""
    traces = sorted(run_dir.glob("train_iteration_*.html"))
    result = {}
    for trace_path in traces:
        step = int(re.search(r"(\d+)", trace_path.stem.split("iteration_")[1]).group(1))
        episodes = parse_trace_file(trace_path)
        if episodes:
            result[step] = episodes
    return result


def group_episodes(episodes: list[dict]) -> list[list[dict]]:
    """Group consecutive episodes by problem text."""
    groups = []
    current_group = []
    current_problem = None
    for ep in episodes:
        if ep["problem"] != current_problem:
            if current_group:
                groups.append(current_group)
            current_group = [ep]
            current_problem = ep["problem"]
        else:
            current_group.append(ep)
    if current_group:
        groups.append(current_group)
    return groups


# ── Analysis ─────────────────────────────────────────────────────────────

def analysis_1_distribution(all_steps: dict[int, list[dict]]):
    """Report actual distribution across ALL episodes."""
    print("=" * 80)
    print("ANALYSIS 1: Real Distribution")
    print("=" * 80)

    all_episodes = []
    for step, eps in sorted(all_steps.items()):
        for ep in eps:
            ep_copy = dict(ep)
            ep_copy["step"] = step
            all_episodes.append(ep_copy)

    total = len(all_episodes)
    cats = ["truncated", "wrong", "correct"]

    print(f"\nTotal episodes: {total}")
    print(f"{'Category':<12} {'Count':>6} {'Frac':>8} {'Mean chars':>12} {'Med chars':>11} {'Std chars':>11} {'Mean tok_est':>13}")
    print("-" * 80)
    for cat in cats:
        subset = [e for e in all_episodes if e["category"] == cat]
        n = len(subset)
        frac = n / total if total else 0
        chars = [e["response_chars"] for e in subset]
        tokens = [e["response_tokens_est"] for e in subset]
        if chars:
            print(f"{cat:<12} {n:>6} {frac:>8.3f} {np.mean(chars):>12.0f} {np.median(chars):>11.0f} {np.std(chars):>11.0f} {np.mean(tokens):>13.0f}")
        else:
            print(f"{cat:<12} {n:>6} {frac:>8.3f} {'N/A':>12} {'N/A':>11} {'N/A':>11} {'N/A':>13}")

    # Early vs late
    steps_sorted = sorted(all_steps.keys())
    n_steps = len(steps_sorted)
    early_steps = steps_sorted[:n_steps // 3]
    late_steps = steps_sorted[2 * n_steps // 3:]

    print(f"\nEarly steps ({early_steps[0]}-{early_steps[-1]}) vs Late steps ({late_steps[0]}-{late_steps[-1]}):")
    print(f"{'Category':<12} {'Early frac':>11} {'Late frac':>10} {'Early mean_chars':>17} {'Late mean_chars':>16}")
    print("-" * 70)
    for cat in cats:
        early_all = [e for e in all_episodes if e["step"] in early_steps]
        late_all = [e for e in all_episodes if e["step"] in late_steps]
        early_cat = [e for e in early_all if e["category"] == cat]
        late_cat = [e for e in late_all if e["category"] == cat]
        ef = len(early_cat) / len(early_all) if early_all else 0
        lf = len(late_cat) / len(late_all) if late_all else 0
        ec = np.mean([e["response_chars"] for e in early_cat]) if early_cat else float("nan")
        lc = np.mean([e["response_chars"] for e in late_cat]) if late_cat else float("nan")
        print(f"{cat:<12} {ef:>11.3f} {lf:>10.3f} {ec:>17.0f} {lc:>16.0f}")


def analysis_2_advantages(all_steps: dict[int, list[dict]]):
    """Compute advantages using real functions for each group."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Advantage Computation on Real Rewards")
    print("=" * 80)

    schemes = ["maxrl", "mean_center"]
    cat_advantages = {s: defaultdict(list) for s in schemes}

    for step, eps in sorted(all_steps.items()):
        groups = group_episodes(eps)
        for group in groups:
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in schemes:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat_advantages[scheme][ep["category"]].append(float(adv))

    cats = ["truncated", "wrong", "correct"]
    for scheme in schemes:
        print(f"\nScheme: {scheme}")
        print(f"{'Category':<12} {'Count':>6} {'Mean adv':>10} {'Std adv':>10} {'Mean |adv|':>11} {'Frac zero':>10}")
        print("-" * 65)
        for cat in cats:
            vals = cat_advantages[scheme][cat]
            if vals:
                arr = np.array(vals)
                nonzero = np.sum(np.abs(arr) > 1e-8)
                print(f"{cat:<12} {len(vals):>6} {np.mean(arr):>10.4f} {np.std(arr):>10.4f} {np.mean(np.abs(arr)):>11.4f} {1 - nonzero/len(arr):>10.3f}")
            else:
                print(f"{cat:<12} {0:>6} {'N/A':>10} {'N/A':>10} {'N/A':>11} {'N/A':>10}")


def analysis_3_gradient_budget(all_steps: dict[int, list[dict]], label: str = ""):
    """Compute gradient budget per category under token_sum vs per_traj_mean."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS 3: Gradient Budget Allocation {label}")
    print("=" * 80)

    schemes = ["maxrl", "mean_center"]
    cats = ["truncated", "wrong", "correct"]

    # Per-step results for tracking evolution
    step_budgets = {scheme: {mode: defaultdict(lambda: defaultdict(float))
                             for mode in ["token_sum", "per_traj_mean"]}
                    for scheme in schemes}

    # Aggregate results
    agg_budgets = {scheme: {mode: defaultdict(float)
                            for mode in ["token_sum", "per_traj_mean"]}
                   for scheme in schemes}

    for step, eps in sorted(all_steps.items()):
        groups = group_episodes(eps)
        for group in groups:
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in schemes:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    tokens = ep["response_tokens_est"]

                    token_budget = abs_adv * tokens
                    traj_budget = abs_adv

                    agg_budgets[scheme]["token_sum"][cat] += token_budget
                    agg_budgets[scheme]["per_traj_mean"][cat] += traj_budget
                    step_budgets[scheme]["token_sum"][step][cat] += token_budget
                    step_budgets[scheme]["per_traj_mean"][step][cat] += traj_budget

    for scheme in schemes:
        print(f"\nScheme: {scheme}")
        for mode in ["token_sum", "per_traj_mean"]:
            budgets = agg_budgets[scheme][mode]
            total = sum(budgets.values())
            print(f"\n  {mode}:")
            print(f"  {'Category':<12} {'Budget':>12} {'Fraction':>10}")
            print(f"  {'-'*36}")
            for cat in cats:
                val = budgets[cat]
                frac = val / total if total else 0
                print(f"  {cat:<12} {val:>12.1f} {frac:>10.3f}")
            print(f"  {'TOTAL':<12} {total:>12.1f} {1.0:>10.3f}")

            # Ratio: truncated / correct
            if budgets["correct"] > 0:
                ratio = budgets["truncated"] / budgets["correct"]
                print(f"  Ratio truncated/correct: {ratio:.3f}")
            else:
                print(f"  Ratio truncated/correct: inf (no correct budget)")

    # Evolution over time
    print("\n  Evolution across steps (ratio truncated/correct):")
    steps_sorted = sorted(all_steps.keys())
    for scheme in schemes:
        print(f"\n  Scheme: {scheme}")
        print(f"  {'Step':>6} {'token_sum ratio':>16} {'per_traj_mean ratio':>20} {'token_sum frac_correct':>23} {'ptm frac_correct':>18}")
        print(f"  {'-'*86}")
        for step in steps_sorted:
            for mode in ["token_sum", "per_traj_mean"]:
                sb = step_budgets[scheme][mode][step]
            # token_sum
            ts_b = step_budgets[scheme]["token_sum"][step]
            ts_total = sum(ts_b.values())
            ts_ratio = ts_b["truncated"] / ts_b["correct"] if ts_b["correct"] > 0 else float("inf")
            ts_fc = ts_b["correct"] / ts_total if ts_total > 0 else 0

            # per_traj_mean
            ptm_b = step_budgets[scheme]["per_traj_mean"][step]
            ptm_total = sum(ptm_b.values())
            ptm_ratio = ptm_b["truncated"] / ptm_b["correct"] if ptm_b["correct"] > 0 else float("inf")
            ptm_fc = ptm_b["correct"] / ptm_total if ptm_total > 0 else 0

            print(f"  {step:>6} {ts_ratio:>16.3f} {ptm_ratio:>20.3f} {ts_fc:>23.3f} {ptm_fc:>18.3f}")

    return agg_budgets, step_budgets


def analysis_4_counterfactual(all_steps: dict[int, list[dict]]):
    """Counterfactual: fraction of gradient signal pointing toward correct responses."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Counterfactual — Fraction of Gradient Signal Toward Correct")
    print("=" * 80)

    schemes = ["maxrl", "mean_center"]
    cats = ["truncated", "wrong", "correct"]
    steps_sorted = sorted(all_steps.keys())

    # Compute per-step budgets for both modes
    results = {scheme: {mode: {} for mode in ["token_sum", "per_traj_mean"]}
               for scheme in schemes}

    for step, eps in sorted(all_steps.items()):
        groups = group_episodes(eps)
        for scheme in schemes:
            for mode in ["token_sum", "per_traj_mean"]:
                if step not in results[scheme][mode]:
                    results[scheme][mode][step] = defaultdict(float)

        for group in groups:
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in schemes:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    tokens = ep["response_tokens_est"]
                    results[scheme]["token_sum"][step][cat] += abs_adv * tokens
                    results[scheme]["per_traj_mean"][step][cat] += abs_adv

    for scheme in schemes:
        print(f"\nScheme: {scheme}")
        print(f"{'Step':>6} {'frac_correct (token_sum)':>26} {'frac_correct (per_traj_mean)':>30} {'Δ (ptm - ts)':>14}")
        print("-" * 80)

        ts_total_c = 0
        ts_total_all = 0
        ptm_total_c = 0
        ptm_total_all = 0

        for step in steps_sorted:
            ts_b = results[scheme]["token_sum"][step]
            ts_total = sum(ts_b.values())
            ts_fc = ts_b["correct"] / ts_total if ts_total > 0 else 0

            ptm_b = results[scheme]["per_traj_mean"][step]
            ptm_total = sum(ptm_b.values())
            ptm_fc = ptm_b["correct"] / ptm_total if ptm_total > 0 else 0

            delta = ptm_fc - ts_fc

            print(f"{step:>6} {ts_fc:>26.4f} {ptm_fc:>30.4f} {delta:>14.4f}")

            ts_total_c += ts_b["correct"]
            ts_total_all += ts_total
            ptm_total_c += ptm_b["correct"]
            ptm_total_all += ptm_total

        # Aggregate
        agg_ts = ts_total_c / ts_total_all if ts_total_all > 0 else 0
        agg_ptm = ptm_total_c / ptm_total_all if ptm_total_all > 0 else 0
        print("-" * 80)
        print(f"{'AGG':>6} {agg_ts:>26.4f} {agg_ptm:>30.4f} {agg_ptm - agg_ts:>14.4f}")


def main():
    base = Path("logs/gpqa-experiment")

    # ── Primary run: gpqa-g8-s42 ──
    print("\n" + "#" * 80)
    print("# PRIMARY RUN: gpqa-g8-s42")
    print("#" * 80)

    run_dir = base / "gpqa-g8-s42"
    all_steps = parse_run(run_dir)
    print(f"\nParsed {len(all_steps)} steps, {sum(len(v) for v in all_steps.values())} total episodes")

    analysis_1_distribution(all_steps)
    analysis_2_advantages(all_steps)
    agg_budgets_primary, step_budgets_primary = analysis_3_gradient_budget(all_steps)
    analysis_4_counterfactual(all_steps)

    # ── Analysis 5: Cross-run consistency ──
    print("\n" + "#" * 80)
    print("# ANALYSIS 5: Cross-Run Consistency")
    print("#" * 80)

    cross_runs = ["gpqa-g16-s42", "gpqa-g8-s137"]
    for run_name in cross_runs:
        run_dir = base / run_name
        if not run_dir.exists():
            print(f"\n  Run {run_name} not found, skipping")
            continue
        print(f"\n{'='*80}")
        print(f"Run: {run_name}")
        print(f"{'='*80}")
        run_steps = parse_run(run_dir)
        print(f"Parsed {len(run_steps)} steps, {sum(len(v) for v in run_steps.values())} total episodes")
        analysis_1_distribution(run_steps)
        analysis_3_gradient_budget(run_steps, label=f"({run_name})")
        analysis_4_counterfactual(run_steps)


if __name__ == "__main__":
    main()
