"""Compute gradient weight ratio: truncated vs correct episodes under token_sum vs per_traj_mean."""

import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinker_cookbook.rl.data_processing import _normalize_subgroup


def parse_trace(html_path: Path) -> list[dict]:
    """Parse a single HTML trace file, returning list of episode dicts grouped by problem."""
    with open(html_path) as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    ps = soup.find_all("p", class_="lt-p")
    episodes = []
    i = 0
    # Skip intro paragraph(s)
    while i < len(ps):
        text = ps[i].get_text().strip()
        if text.startswith("Problem:"):
            break
        i += 1

    while i < len(ps) - 2:
        prob_text = ps[i].get_text().strip()
        if not prob_text.startswith("Problem:"):
            i += 1
            continue
        resp_text = ps[i + 1].get_text().strip()
        # ref_text = ps[i + 2].get_text().strip()  # unused
        meta_text = ps[i + 3].get_text().strip()

        # Parse metadata: "Boxed: ✓, EOS: ✓, Correct: ✓, Reward: 1.00"
        m = re.match(
            r"Boxed: ([✓✗]), EOS: ([✓✗]), Correct: ([✓✗]), Reward: ([\d.-]+)",
            meta_text,
        )
        if not m:
            i += 1
            continue

        resp_len = len(resp_text) - len("Response: ")  # strip prefix
        episodes.append(
            {
                "problem": prob_text[:200],  # truncate for grouping key
                "resp_len": max(resp_len, 1),
                "eos": m.group(2) == "✓",
                "correct": m.group(3) == "✓",
                "reward": float(m.group(4)),
            }
        )
        i += 4

    return episodes


def group_episodes(episodes: list[dict]) -> list[list[dict]]:
    """Group consecutive episodes with the same problem text."""
    groups = []
    current = []
    for ep in episodes:
        if current and ep["problem"] != current[0]["problem"]:
            groups.append(current)
            current = []
        current.append(ep)
    if current:
        groups.append(current)
    return groups


def compute_weights(groups, scheme, alpha=0.5):
    """Compute per-category gradient weight sums across all groups.

    Returns dict: {category: {"token_sum": float, "ptm": float}}
    Categories: truncated, wrong, correct
    """
    totals = defaultdict(lambda: {"token_sum": 0.0, "ptm": 0.0})

    for group in groups:
        rewards = torch.tensor([ep["reward"] for ep in group])
        advs = _normalize_subgroup(rewards, scheme=scheme, alpha=alpha)

        for ep, adv_val in zip(group, advs.tolist()):
            abs_adv = abs(adv_val)
            if not ep["eos"]:
                cat = "truncated"
            elif ep["correct"]:
                cat = "correct"
            else:
                cat = "wrong"

            totals[cat]["token_sum"] += abs_adv * ep["resp_len"]
            totals[cat]["ptm"] += abs_adv

    return dict(totals)


def ratio_str(num, den):
    if den < 1e-12:
        return "inf" if num > 1e-12 else "n/a"
    return f"{num / den:.3f}"


def analyze_run(run_dir: Path, label: str):
    files = sorted(run_dir.glob("train_iteration_*.html"))
    print(f"\n{'='*90}")
    print(f"  {label}: {run_dir.name} ({len(files)} steps, 4 groups sampled per step)")
    print(f"{'='*90}")

    all_groups = []
    step_data = []

    for fpath in files:
        step = int(fpath.stem.split("_")[-1])
        episodes = parse_trace(fpath)
        groups = group_episodes(episodes)
        all_groups.extend(groups)

        n_trunc = sum(1 for g in groups for ep in g if not ep["eos"])
        n_correct = sum(1 for g in groups for ep in g if ep["eos"] and ep["correct"])
        n_wrong = sum(1 for g in groups for ep in g if ep["eos"] and not ep["correct"])
        n_total = sum(len(g) for g in groups)

        w_maxrl = compute_weights(groups, "maxrl", 0.5)
        w_mc = compute_weights(groups, "mean_center", 0.5)

        step_data.append((step, n_total, n_trunc, n_correct, n_wrong, w_maxrl, w_mc))

    # Print episode counts per step
    print(f"\n  Episode counts per step (sampled subset):")
    print(f"  {'step':>4}  {'total':>5}  {'trunc':>5}  {'correct':>7}  {'wrong':>5}")
    for step, n_total, n_trunc, n_correct, n_wrong, _, _ in step_data:
        print(f"  {step:>4}  {n_total:>5}  {n_trunc:>5}  {n_correct:>7}  {n_wrong:>5}")

    # Print ratio table
    header = (
        f"  {'step':>4}  "
        f"{'trunc/corr':>10} {'trunc/corr':>10}  "
        f"{'trunc/corr':>10} {'trunc/corr':>10}"
    )
    subhdr = (
        f"  {'':>4}  "
        f"{'tkn_sum':>10} {'ptm':>10}  "
        f"{'tkn_sum':>10} {'ptm':>10}"
    )
    print(f"\n  Gradient weight ratio: truncated / correct")
    print(f"  {'':>4}  {'--- maxrl ---':>21}  {'--- mean_center ---':>21}")
    print(header)
    print(subhdr)
    print(f"  {'-'*56}")

    for step, _, _, _, _, w_maxrl, w_mc in step_data:
        vals = []
        for w in [w_maxrl, w_mc]:
            t_ts = w.get("truncated", {}).get("token_sum", 0)
            c_ts = w.get("correct", {}).get("token_sum", 0)
            t_pt = w.get("truncated", {}).get("ptm", 0)
            c_pt = w.get("correct", {}).get("ptm", 0)
            vals.extend([ratio_str(t_ts, c_ts), ratio_str(t_pt, c_pt)])
        print(f"  {step:>4}  {vals[0]:>10} {vals[1]:>10}  {vals[2]:>10} {vals[3]:>10}")

    # Aggregate
    w_maxrl_all = compute_weights(all_groups, "maxrl", 0.5)
    w_mc_all = compute_weights(all_groups, "mean_center", 0.5)

    print(f"\n  Aggregate raw budget:")
    for label2, w in [("maxrl", w_maxrl_all), ("mean_center", w_mc_all)]:
        print(f"\n    {label2}:")
        for cat in ["truncated", "correct", "wrong"]:
            d = w.get(cat, {"token_sum": 0, "ptm": 0})
            print(f"      {cat:>10}: token_sum={d['token_sum']:>12.1f}  ptm={d['ptm']:>8.3f}")

    print(f"\n  Aggregate ratios (truncated / correct):")
    for label2, w in [("maxrl", w_maxrl_all), ("mean_center", w_mc_all)]:
        t_ts = w.get("truncated", {}).get("token_sum", 0)
        c_ts = w.get("correct", {}).get("token_sum", 0)
        t_pt = w.get("truncated", {}).get("ptm", 0)
        c_pt = w.get("correct", {}).get("ptm", 0)
        print(
            f"    {label2:>12}: token_sum={ratio_str(t_ts, c_ts):>8}  ptm={ratio_str(t_pt, c_pt):>8}"
        )

    print(f"\n  NOTE: Only 4/{64 if 'g8' in run_dir.name else 64} groups sampled per step. Ratios are representative but not exhaustive.")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "logs" / "gpqa-experiment"
    analyze_run(base / "gpqa-g8-s42", "G=8, seed=42")
    analyze_run(base / "gpqa-g16-s42", "G=16, seed=42")
