"""
Definitive gradient budget allocation analysis from ALL available GPQA RLVR training data.

Uses ALL 6 runs (3 seeds × 2 group sizes) with real advantage functions.
Outputs to stdout AND logs/gradient_budget_results.txt.
"""

import json
import re
import sys
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from bs4 import BeautifulSoup

# Import the real advantage function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinker_cookbook.rl.data_processing import _normalize_subgroup


# ── Config ────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent / "logs" / "gpqa-experiment"
RUNS = {
    "g8": ["gpqa-g8-s42", "gpqa-g8-s137", "gpqa-g8-s7"],
    "g16": ["gpqa-g16-s42", "gpqa-g16-s137", "gpqa-g16-s7"],
}
ALL_RUNS = [r for runs in RUNS.values() for r in runs]
METRICS_KEYS = [
    "env/all/correct",
    "env/all/format_boxed",
    "env/all/format_eos",
    "env/all/ac_tokens_per_turn",
    "env/all/reward/total",
    "env/all/by_group/frac_all_bad",
    "env/all/by_group/frac_all_good",
    "env/all/by_group/frac_mixed",
    "optim/entropy",
]
CATS = ["truncated", "wrong", "correct"]
SCHEMES = ["maxrl", "mean_center"]
MODES = ["token_sum", "per_traj_mean"]
BOOTSTRAP_N = 1000


# ── Tee output ────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


# ── Parsing ───────────────────────────────────────────────────────────────

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
            if i + 1 >= len(ps):
                i += 1; continue
            response_text = ps[i + 1].get_text().strip()
            if i + 2 >= len(ps):
                i += 1; continue
            ref_text = ps[i + 2].get_text().strip()
            if i + 3 >= len(ps):
                i += 1; continue
            score_text = ps[i + 3].get_text().strip()

            if not score_text.startswith("Boxed:"):
                i += 1; continue

            boxed = "\u2713" in score_text.split("Boxed:")[1].split(",")[0]
            eos = "\u2713" in score_text.split("EOS:")[1].split(",")[0]
            correct = "\u2713" in score_text.split("Correct:")[1].split(",")[0]
            reward_match = re.search(r"Reward:\s*([-\d.]+)", score_text)
            reward = float(reward_match.group(1)) if reward_match else 0.0

            resp = response_text
            if resp.startswith("Response:"):
                resp = resp[len("Response:"):].strip()
            resp_chars = len(resp)

            if not eos:
                category = "truncated"
            elif not correct:
                category = "wrong"
            else:
                category = "correct"

            episodes.append({
                "problem": problem_text[:200],
                "response_chars": resp_chars,
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
    """Parse all train_iteration_*.html files. Returns {step: [episodes]}."""
    traces = sorted(run_dir.glob("train_iteration_*.html"))
    result = {}
    for trace_path in traces:
        step = int(re.search(r"(\d+)", trace_path.stem.split("iteration_")[1]).group(1))
        episodes = parse_trace_file(trace_path)
        if episodes:
            result[step] = episodes
    return result


def group_episodes(episodes: list[dict], expected_g: int | None = None) -> list[list[dict]]:
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
    # Validate group sizes if expected
    if expected_g is not None:
        for g in groups:
            if len(g) != expected_g:
                print(f"  WARNING: group size {len(g)} != expected {expected_g}")
    return groups


def load_metrics(run_dir: Path) -> list[dict]:
    """Load metrics.jsonl as list of dicts."""
    path = run_dir / "metrics.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── Part 1: Full-batch statistics from metrics.jsonl ──────────────────────

def part1_metrics():
    print("\n" + "=" * 100)
    print("PART 1: Full-Batch Statistics from metrics.jsonl")
    print("=" * 100)

    all_metrics = {}  # run_name -> list of step dicts
    for run_name in ALL_RUNS:
        all_metrics[run_name] = load_metrics(BASE / run_name)

    # 1a. Per-run summary
    print("\n--- 1a. Per-run summary: start, end, slope ---")
    for run_name in ALL_RUNS:
        metrics = all_metrics[run_name]
        steps = [m["step"] for m in metrics]
        n_steps = len(steps)
        print(f"\n  Run: {run_name} ({n_steps} steps, step range {steps[0]}-{steps[-1]})")
        print(f"  {'Metric':<40} {'Start':>8} {'End':>8} {'Slope/step':>12}")
        print(f"  {'-'*70}")
        for key in METRICS_KEYS:
            vals = [m.get(key) for m in metrics]
            if vals[0] is None:
                continue
            start = vals[0]
            end = vals[-1]
            # Linear regression slope
            x = np.array(steps, dtype=float)
            y = np.array(vals, dtype=float)
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            print(f"  {key:<40} {start:>8.4f} {end:>8.4f} {slope:>12.6f}")

    # 1b. Cross-run mean ± std at each step, separated by G
    print("\n\n--- 1b. Cross-run mean +/- std at each step ---")
    for g_label, run_names in RUNS.items():
        print(f"\n  Group: {g_label}")
        metrics_list = [all_metrics[r] for r in run_names]

        # Find common steps
        step_sets = [set(m["step"] for m in ms) for ms in metrics_list]
        common_steps = sorted(set.intersection(*step_sets))

        for key in METRICS_KEYS:
            print(f"\n    {key}:")
            print(f"    {'Step':>6}  {'Mean':>8}  {'Std':>8}  {'Values':>30}")
            print(f"    {'-'*60}")
            for step in common_steps:
                vals = []
                for ms in metrics_list:
                    step_dict = next((m for m in ms if m["step"] == step), None)
                    if step_dict and key in step_dict:
                        vals.append(step_dict[key])
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    vals_str = ", ".join(f"{v:.4f}" for v in vals)
                    print(f"    {step:>6}  {mean:>8.4f}  {std:>8.4f}  {vals_str}")

    # 1c. Max deviation from mean within each G
    print("\n\n--- 1c. Max deviation from mean within each G ---")
    for g_label, run_names in RUNS.items():
        print(f"\n  Group: {g_label}")
        metrics_list = [all_metrics[r] for r in run_names]
        step_sets = [set(m["step"] for m in ms) for ms in metrics_list]
        common_steps = sorted(set.intersection(*step_sets))

        print(f"  {'Metric':<40} {'Max deviation':>14} {'At step':>8} {'Mean at that step':>18}")
        print(f"  {'-'*82}")
        for key in METRICS_KEYS:
            max_dev = 0.0
            max_dev_step = -1
            max_dev_mean = 0.0
            for step in common_steps:
                vals = []
                for ms in metrics_list:
                    step_dict = next((m for m in ms if m["step"] == step), None)
                    if step_dict and key in step_dict:
                        vals.append(step_dict[key])
                if len(vals) >= 2:
                    mean = np.mean(vals)
                    dev = max(abs(v - mean) for v in vals)
                    if dev > max_dev:
                        max_dev = dev
                        max_dev_step = step
                        max_dev_mean = mean
            print(f"  {key:<40} {max_dev:>14.6f} {max_dev_step:>8} {max_dev_mean:>18.4f}")


# ── Part 2: Gradient Budget from Trace Files ─────────────────────────────

def parse_all_runs() -> dict[str, dict[int, list[dict]]]:
    """Parse ALL trace files from ALL runs. Returns {run_name: {step: [episodes]}}."""
    all_data = {}
    for run_name in ALL_RUNS:
        run_dir = BASE / run_name
        all_data[run_name] = parse_run(run_dir)
        n_eps = sum(len(v) for v in all_data[run_name].values())
        n_steps = len(all_data[run_name])
        print(f"  Parsed {run_name}: {n_steps} steps, {n_eps} episodes")
    return all_data


def get_g_size(run_name: str) -> int:
    return 16 if "g16" in run_name else 8


def compute_group_budgets(groups: list[list[dict]]) -> dict:
    """For a list of groups, compute gradient budgets under all scheme×mode combos.

    Returns dict with:
      - per_group: list of dicts with per-group info
      - aggregate: {scheme: {mode: {cat: budget}}}
    """
    per_group = []
    agg = {s: {m: defaultdict(float) for m in MODES} for s in SCHEMES}

    for group in groups:
        rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
        group_info = {"episodes": group, "rewards": rewards.tolist()}
        group_budgets = {}

        for scheme in SCHEMES:
            advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
            group_budgets[scheme] = {}

            for mode in MODES:
                cat_budget = defaultdict(float)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    if mode == "token_sum":
                        budget = abs_adv * (ep["response_chars"] / 4.0)  # token estimate
                    else:  # per_traj_mean
                        budget = abs_adv
                    cat_budget[cat] += budget
                    agg[scheme][mode][cat] += budget
                group_budgets[scheme][mode] = dict(cat_budget)

        group_info["budgets"] = group_budgets
        per_group.append(group_info)

    return {"per_group": per_group, "aggregate": agg}


def phase_label(step: int) -> str:
    if step <= 5:
        return "early"
    elif step <= 12:
        return "mid"
    else:
        return "late"


def part2_gradient_budget(all_data: dict[str, dict[int, list[dict]]]):
    print("\n" + "=" * 100)
    print("PART 2: Gradient Budget from Trace Files")
    print("=" * 100)

    # Count total episodes
    total_episodes = 0
    total_groups = 0
    for run_name, steps in all_data.items():
        g_size = get_g_size(run_name)
        for step, eps in steps.items():
            total_episodes += len(eps)
            groups = group_episodes(eps, expected_g=g_size)
            total_groups += len(groups)

    print(f"\n  Total episodes parsed: {total_episodes}")
    print(f"  Total groups: {total_groups}")
    expected = 3 * 20 * 4 * 8 + 3 * 15 * 4 * 16  # G=8 + G=16
    print(f"  Expected: ~{expected} episodes (actual may vary due to parsing)")

    # Aggregate across ALL runs
    all_groups = []
    for run_name, steps in all_data.items():
        g_size = get_g_size(run_name)
        for step, eps in steps.items():
            groups = group_episodes(eps, expected_g=g_size)
            for g in groups:
                all_groups.append({"group": g, "run": run_name, "step": step, "g_size": g_size})

    print(f"\n  Total groups for budget analysis: {len(all_groups)}")

    # ── Main budget table ──
    print("\n--- 2a. Aggregate Gradient Budget (ALL runs) ---")
    agg = {s: {m: defaultdict(float) for m in MODES} for s in SCHEMES}

    for item in all_groups:
        group = item["group"]
        rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
        for scheme in SCHEMES:
            advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
            for ep, adv in zip(group, advs):
                cat = ep["category"]
                abs_adv = abs(float(adv))
                agg[scheme]["token_sum"][cat] += abs_adv * (ep["response_chars"] / 4.0)
                agg[scheme]["per_traj_mean"][cat] += abs_adv

    print(f"\n  {'Scheme x Mode':<30} {'trunc':>10} {'wrong':>10} {'correct':>10} {'trunc/corr':>12}")
    print(f"  {'-'*74}")
    for scheme in SCHEMES:
        for mode in MODES:
            b = agg[scheme][mode]
            total = sum(b[c] for c in CATS)
            fracs = {c: b[c] / total if total > 0 else 0 for c in CATS}
            ratio = b["truncated"] / b["correct"] if b["correct"] > 0 else float("inf")
            label = f"{scheme} x {mode}"
            print(f"  {label:<30} {fracs['truncated']:>10.4f} {fracs['wrong']:>10.4f} {fracs['correct']:>10.4f} {ratio:>12.3f}")

    # ── Breakdown by G=8 vs G=16 ──
    print("\n--- 2b. Budget breakdown: G=8 vs G=16 ---")
    for g_val in [8, 16]:
        print(f"\n  G={g_val}:")
        sub_agg = {s: {m: defaultdict(float) for m in MODES} for s in SCHEMES}
        for item in all_groups:
            if item["g_size"] != g_val:
                continue
            group = item["group"]
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in SCHEMES:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    sub_agg[scheme]["token_sum"][cat] += abs_adv * (ep["response_chars"] / 4.0)
                    sub_agg[scheme]["per_traj_mean"][cat] += abs_adv

        print(f"  {'Scheme x Mode':<30} {'trunc':>10} {'wrong':>10} {'correct':>10} {'trunc/corr':>12}")
        print(f"  {'-'*74}")
        for scheme in SCHEMES:
            for mode in MODES:
                b = sub_agg[scheme][mode]
                total = sum(b[c] for c in CATS)
                fracs = {c: b[c] / total if total > 0 else 0 for c in CATS}
                ratio = b["truncated"] / b["correct"] if b["correct"] > 0 else float("inf")
                label = f"{scheme} x {mode}"
                print(f"  {label:<30} {fracs['truncated']:>10.4f} {fracs['wrong']:>10.4f} {fracs['correct']:>10.4f} {ratio:>12.3f}")

    # ── Breakdown by phase ──
    print("\n--- 2c. Budget breakdown: early (0-5) vs mid (6-12) vs late (13+) ---")
    for phase in ["early", "mid", "late"]:
        print(f"\n  Phase: {phase}")
        sub_agg = {s: {m: defaultdict(float) for m in MODES} for s in SCHEMES}
        n_groups = 0
        for item in all_groups:
            if phase_label(item["step"]) != phase:
                continue
            n_groups += 1
            group = item["group"]
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in SCHEMES:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    sub_agg[scheme]["token_sum"][cat] += abs_adv * (ep["response_chars"] / 4.0)
                    sub_agg[scheme]["per_traj_mean"][cat] += abs_adv

        print(f"  Groups in phase: {n_groups}")
        print(f"  {'Scheme x Mode':<30} {'trunc':>10} {'wrong':>10} {'correct':>10} {'trunc/corr':>12}")
        print(f"  {'-'*74}")
        for scheme in SCHEMES:
            for mode in MODES:
                b = sub_agg[scheme][mode]
                total = sum(b[c] for c in CATS)
                fracs = {c: b[c] / total if total > 0 else 0 for c in CATS}
                ratio = b["truncated"] / b["correct"] if b["correct"] > 0 else float("inf")
                label = f"{scheme} x {mode}"
                print(f"  {label:<30} {fracs['truncated']:>10.4f} {fracs['wrong']:>10.4f} {fracs['correct']:>10.4f} {ratio:>12.3f}")

    # ── Per seed ──
    print("\n--- 2d. Budget breakdown: per seed ---")
    for run_name in ALL_RUNS:
        print(f"\n  Run: {run_name}")
        sub_agg = {s: {m: defaultdict(float) for m in MODES} for s in SCHEMES}
        n_groups = 0
        for item in all_groups:
            if item["run"] != run_name:
                continue
            n_groups += 1
            group = item["group"]
            rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
            for scheme in SCHEMES:
                advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    sub_agg[scheme]["token_sum"][cat] += abs_adv * (ep["response_chars"] / 4.0)
                    sub_agg[scheme]["per_traj_mean"][cat] += abs_adv

        print(f"  Groups: {n_groups}")
        print(f"  {'Scheme x Mode':<30} {'trunc':>10} {'wrong':>10} {'correct':>10} {'trunc/corr':>12}")
        print(f"  {'-'*74}")
        for scheme in SCHEMES:
            for mode in MODES:
                b = sub_agg[scheme][mode]
                total = sum(b[c] for c in CATS)
                fracs = {c: b[c] / total if total > 0 else 0 for c in CATS}
                ratio = b["truncated"] / b["correct"] if b["correct"] > 0 else float("inf")
                label = f"{scheme} x {mode}"
                print(f"  {label:<30} {fracs['truncated']:>10.4f} {fracs['wrong']:>10.4f} {fracs['correct']:>10.4f} {ratio:>12.3f}")

    return all_groups


# ── Part 3: Bootstrap Confidence Intervals ────────────────────────────────

def part3_bootstrap(all_groups: list[dict]):
    print("\n" + "=" * 100)
    print("PART 3: Bootstrap 95% CI for trunc/correct Ratio")
    print("=" * 100)

    rng = np.random.default_rng(42)

    # Pre-compute per-group budget contributions
    group_budgets = []  # list of {scheme: {mode: {cat: budget}}}
    for item in all_groups:
        group = item["group"]
        rewards = torch.tensor([e["reward"] for e in group], dtype=torch.float32)
        gb = {}
        for scheme in SCHEMES:
            advs = _normalize_subgroup(rewards, scheme, alpha=0.5)
            gb[scheme] = {}
            for mode in MODES:
                cat_budget = defaultdict(float)
                for ep, adv in zip(group, advs):
                    cat = ep["category"]
                    abs_adv = abs(float(adv))
                    if mode == "token_sum":
                        cat_budget[cat] += abs_adv * (ep["response_chars"] / 4.0)
                    else:
                        cat_budget[cat] += abs_adv
                gb[scheme][mode] = dict(cat_budget)
        group_budgets.append(gb)

    n = len(group_budgets)

    print(f"\n  Number of groups: {n}")
    print(f"  Bootstrap iterations: {BOOTSTRAP_N}")

    for scheme in SCHEMES:
        for mode in MODES:
            label = f"{scheme} x {mode}"

            # Point estimate
            trunc_total = sum(gb[scheme][mode].get("truncated", 0) for gb in group_budgets)
            correct_total = sum(gb[scheme][mode].get("correct", 0) for gb in group_budgets)
            point_ratio = trunc_total / correct_total if correct_total > 0 else float("inf")

            # Point estimate fractions
            total_all = sum(sum(gb[scheme][mode].get(c, 0) for c in CATS) for gb in group_budgets)
            point_fracs = {c: sum(gb[scheme][mode].get(c, 0) for gb in group_budgets) / total_all for c in CATS}

            # Bootstrap
            boot_ratios = []
            boot_fracs = {c: [] for c in CATS}
            for _ in range(BOOTSTRAP_N):
                idx = rng.integers(0, n, size=n)
                b_trunc = sum(group_budgets[i][scheme][mode].get("truncated", 0) for i in idx)
                b_correct = sum(group_budgets[i][scheme][mode].get("correct", 0) for i in idx)
                b_total = sum(
                    sum(group_budgets[i][scheme][mode].get(c, 0) for c in CATS)
                    for i in idx
                )
                if b_correct > 0:
                    boot_ratios.append(b_trunc / b_correct)
                for c in CATS:
                    bc = sum(group_budgets[i][scheme][mode].get(c, 0) for i in idx)
                    boot_fracs[c].append(bc / b_total if b_total > 0 else 0)

            boot_ratios = np.array(boot_ratios)
            lo_ratio = np.percentile(boot_ratios, 2.5) if len(boot_ratios) > 0 else float("nan")
            hi_ratio = np.percentile(boot_ratios, 97.5) if len(boot_ratios) > 0 else float("nan")

            print(f"\n  {label}:")
            print(f"    trunc/correct ratio: {point_ratio:.4f}  95% CI: [{lo_ratio:.4f}, {hi_ratio:.4f}]")
            for c in CATS:
                arr = np.array(boot_fracs[c])
                lo = np.percentile(arr, 2.5)
                hi = np.percentile(arr, 97.5)
                print(f"    frac_{c}: {point_fracs[c]:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")


# ── Part 4: Reward Distribution Verification ─────────────────────────────

def part4_reward_distribution(all_data: dict[str, dict[int, list[dict]]]):
    print("\n" + "=" * 100)
    print("PART 4: Reward Distribution Verification")
    print("=" * 100)

    all_episodes = []
    for run_name, steps in all_data.items():
        for step, eps in steps.items():
            for ep in eps:
                ep_copy = dict(ep)
                ep_copy["run"] = run_name
                ep_copy["step"] = step
                all_episodes.append(ep_copy)

    total = len(all_episodes)
    print(f"\n  Total episodes: {total}")

    # 4a. Exact reward values
    print("\n--- 4a. Exact reward values observed ---")
    reward_counts = defaultdict(int)
    for ep in all_episodes:
        reward_counts[ep["reward"]] += 1
    print(f"  {'Reward':>10} {'Count':>8} {'Fraction':>10}")
    print(f"  {'-'*30}")
    for r in sorted(reward_counts.keys()):
        print(f"  {r:>10.4f} {reward_counts[r]:>8} {reward_counts[r]/total:>10.4f}")

    # 4b. Per-category distributions
    print("\n--- 4b. Per-category length distributions (chars) ---")
    print(f"  {'Category':<12} {'N':>6} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*68}")
    for cat in CATS:
        subset = [e for e in all_episodes if e["category"] == cat]
        n = len(subset)
        if n == 0:
            print(f"  {cat:<12} {0:>6} {'N/A':>10}")
            continue
        chars = np.array([e["response_chars"] for e in subset])
        print(f"  {cat:<12} {n:>6} {np.mean(chars):>10.0f} {np.median(chars):>10.0f} {np.std(chars):>10.0f} {np.min(chars):>8} {np.max(chars):>8}")

    # 4c. Per-category length distributions (token estimate)
    print("\n--- 4c. Per-category length distributions (token estimate = chars/4) ---")
    print(f"  {'Category':<12} {'N':>6} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*68}")
    for cat in CATS:
        subset = [e for e in all_episodes if e["category"] == cat]
        n = len(subset)
        if n == 0:
            print(f"  {cat:<12} {0:>6} {'N/A':>10}")
            continue
        tokens = np.array([e["response_chars"] / 4.0 for e in subset])
        print(f"  {cat:<12} {n:>6} {np.mean(tokens):>10.0f} {np.median(tokens):>10.0f} {np.std(tokens):>10.0f} {int(np.min(tokens)):>8} {int(np.max(tokens)):>8}")

    # 4d. Reward by category cross-tab
    print("\n--- 4d. Reward values by category ---")
    for cat in CATS:
        subset = [e for e in all_episodes if e["category"] == cat]
        rc = defaultdict(int)
        for e in subset:
            rc[e["reward"]] += 1
        vals_str = ", ".join(f"{r:.2f}: {c}" for r, c in sorted(rc.items()))
        print(f"  {cat:<12}: {vals_str}")

    # 4e. Per-run category fractions
    print("\n--- 4e. Category fractions per run ---")
    print(f"  {'Run':<20} {'N':>6} {'trunc':>8} {'wrong':>8} {'correct':>8}")
    print(f"  {'-'*54}")
    for run_name in ALL_RUNS:
        subset = [e for e in all_episodes if e["run"] == run_name]
        n = len(subset)
        for cat in CATS:
            cat_n = len([e for e in subset if e["category"] == cat])
        fracs = {c: len([e for e in subset if e["category"] == c]) / n for c in CATS}
        print(f"  {run_name:<20} {n:>6} {fracs['truncated']:>8.3f} {fracs['wrong']:>8.3f} {fracs['correct']:>8.3f}")

    # 4f. Length ratio: truncated / correct
    print("\n--- 4f. Length ratio (truncated mean / correct mean) ---")
    for run_name in ALL_RUNS:
        subset = [e for e in all_episodes if e["run"] == run_name]
        trunc_chars = [e["response_chars"] for e in subset if e["category"] == "truncated"]
        correct_chars = [e["response_chars"] for e in subset if e["category"] == "correct"]
        if trunc_chars and correct_chars:
            ratio = np.mean(trunc_chars) / np.mean(correct_chars)
            print(f"  {run_name:<20}: {ratio:.2f}x  (trunc mean={np.mean(trunc_chars):.0f}, correct mean={np.mean(correct_chars):.0f})")
        else:
            print(f"  {run_name:<20}: N/A")

    # Overall
    trunc_chars = [e["response_chars"] for e in all_episodes if e["category"] == "truncated"]
    correct_chars = [e["response_chars"] for e in all_episodes if e["category"] == "correct"]
    if trunc_chars and correct_chars:
        ratio = np.mean(trunc_chars) / np.mean(correct_chars)
        print(f"  {'ALL':<20}: {ratio:.2f}x  (trunc mean={np.mean(trunc_chars):.0f}, correct mean={np.mean(correct_chars):.0f})")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    output_path = Path(__file__).resolve().parent.parent / "logs" / "gradient_budget_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(output_path, "w")
    sys.stdout = Tee(sys.__stdout__, f)

    print("=" * 100)
    print("DEFINITIVE GRADIENT BUDGET ANALYSIS — ALL 6 GPQA RLVR RUNS")
    print("=" * 100)
    print(f"Runs: {', '.join(ALL_RUNS)}")
    print(f"Base: {BASE}")

    # Part 1
    part1_metrics()

    # Parse all trace files
    print("\n\nParsing all trace files...")
    all_data = parse_all_runs()

    # Part 2
    all_groups = part2_gradient_budget(all_data)

    # Part 3
    part3_bootstrap(all_groups)

    # Part 4
    part4_reward_distribution(all_data)

    # Final summary
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
Key question: Under token_sum weighting, how much gradient budget goes to
truncated responses vs correct responses?

The trunc/correct ratio under token_sum tells us: for every unit of gradient
pushing toward correct answers, how many units push toward truncation-related
patterns (which are systematically longer due to hitting the token limit).

Under per_traj_mean, each trajectory contributes equal gradient regardless of
length, so the length bias is removed. The ratio difference between these two
modes quantifies the length bias.
""")

    f.close()
    sys.stdout = sys.__stdout__
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
