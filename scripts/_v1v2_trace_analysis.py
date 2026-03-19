"""Compare RLVR training transcripts: v1 (token_sum) vs v2 (normalize_advantages_by_length)."""

import glob
import re
import html
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
import numpy as np
from scipy import stats

V1_DIR = "logs/gpqa-experiment/gpqa-g8-s42"
V2_DIR = "logs/gpqa-experiment-v2/gpqa-g8-s42"


def parse_trace_file(filepath: str) -> list[dict]:
    """Parse a single HTML trace file into a list of episode records."""
    step_match = re.search(r"iteration_(\d+)", filepath)
    step = int(step_match.group(1)) if step_match else -1

    with open(filepath) as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    paragraphs = soup.find_all("p", class_="lt-p")

    episodes = []
    i = 0
    while i < len(paragraphs):
        text = paragraphs[i].get_text(strip=False)

        if text.startswith("Problem:") or text.startswith("\nProblem:"):
            problem_text = text.strip()
            # Get question ID (first 80 chars after "Problem: ")
            q_raw = problem_text.replace("Problem: ", "", 1).replace("Problem:", "", 1).strip()
            question_id = q_raw[:80]

            # Next paragraph should be the response
            if i + 1 < len(paragraphs):
                response_text = paragraphs[i + 1].get_text(strip=False).strip()
                if response_text.startswith("Response:"):
                    response_body = response_text[len("Response:"):].strip()
                else:
                    response_body = response_text
            else:
                response_body = ""

            # Next should be reference answer
            ref_answer = ""
            if i + 2 < len(paragraphs):
                ref_text = paragraphs[i + 2].get_text(strip=False).strip()
                if ref_text.startswith("Reference Answer:"):
                    ref_answer = ref_text.replace("Reference Answer:", "").strip()

            # Next should be the status line
            status_line = ""
            if i + 3 < len(paragraphs):
                status_text = paragraphs[i + 3].get_text(strip=False).strip()
                if "Boxed:" in status_text:
                    status_line = status_text

            # Parse status
            format_boxed = "✓" in status_line.split("Boxed:")[1].split(",")[0] if "Boxed:" in status_line else False
            eos = "✓" in status_line.split("EOS:")[1].split(",")[0] if "EOS:" in status_line else False
            correct = "✓" in status_line.split("Correct:")[1].split(",")[0] if "Correct:" in status_line else False

            reward_match = re.search(r"Reward:\s*([\d.]+)", status_line)
            reward = float(reward_match.group(1)) if reward_match else 0.0

            episodes.append({
                "step": step,
                "question_id": question_id,
                "question_full": q_raw,
                "response": response_body,
                "response_len": len(response_body),
                "ref_answer": ref_answer,
                "format_boxed": format_boxed,
                "eos": eos,
                "correct": correct,
                "reward": reward,
            })
            i += 4
        else:
            i += 1

    return episodes


def parse_all_traces(base_dir: str) -> list[dict]:
    """Parse all trace files in a directory."""
    files = sorted(glob.glob(f"{base_dir}/train_iteration_*.html"))
    all_episodes = []
    for f in files:
        eps = parse_trace_file(f)
        all_episodes.extend(eps)
        print(f"  {Path(f).name}: {len(eps)} episodes")
    return all_episodes


def main():
    print("=" * 80)
    print("RLVR Transcript Comparison: v1 (token_sum) vs v2 (normalize_by_length)")
    print("=" * 80)

    print("\n--- Parsing v1 traces ---")
    v1_episodes = parse_all_traces(V1_DIR)
    print(f"\nTotal v1 episodes: {len(v1_episodes)}")

    print("\n--- Parsing v2 traces ---")
    v2_episodes = parse_all_traces(V2_DIR)
    print(f"\nTotal v2 episodes: {len(v2_episodes)}")

    # =========================================================================
    # 1. Per-step summary for ALL steps
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: Per-Step Summary (ALL available steps)")
    print("=" * 80)

    def step_summary(episodes):
        by_step = defaultdict(list)
        for ep in episodes:
            by_step[ep["step"]].append(ep)
        return by_step

    v1_by_step = step_summary(v1_episodes)
    v2_by_step = step_summary(v2_episodes)

    def print_step_table(label, by_step):
        print(f"\n{label}:")
        print(f"{'Step':>5} | {'N':>4} | {'Accuracy':>8} | {'Format%':>8} | {'Trunc%':>8} | {'MeanLen':>8}")
        print("-" * 60)
        for step in sorted(by_step.keys()):
            eps = by_step[step]
            n = len(eps)
            acc = np.mean([e["correct"] for e in eps])
            fmt = np.mean([e["format_boxed"] for e in eps])
            trunc = np.mean([not e["eos"] for e in eps])
            ml = np.mean([e["response_len"] for e in eps])
            print(f"{step:>5} | {n:>4} | {acc:>8.3f} | {fmt:>8.3f} | {trunc:>8.3f} | {ml:>8.0f}")

    print_step_table("v1 (token_sum, no adv normalization)", v1_by_step)
    print_step_table("v2 (normalize_advantages_by_length)", v2_by_step)

    # =========================================================================
    # 2. Matched steps comparison (steps 0, 1, 2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Matched Steps Comparison (steps 0, 1, 2)")
    print("=" * 80)

    matched_steps = sorted(set(v1_by_step.keys()) & set(v2_by_step.keys()))
    print(f"\nMatched steps: {matched_steps}")
    print(f"\nCAVEAT: These are from the trace sample (4 of {8} groups logged per batch).")
    print("Not the full batch. Small-N statistics apply.\n")

    print(f"{'Step':>5} | {'Metric':>12} | {'v1':>8} | {'v2':>8} | {'Delta':>8}")
    print("-" * 55)
    for step in matched_steps:
        v1e = v1_by_step[step]
        v2e = v2_by_step[step]
        for metric_name, metric_fn in [
            ("Accuracy", lambda eps: np.mean([e["correct"] for e in eps])),
            ("Format%", lambda eps: np.mean([e["format_boxed"] for e in eps])),
            ("Trunc%", lambda eps: np.mean([not e["eos"] for e in eps])),
            ("MeanLen", lambda eps: np.mean([e["response_len"] for e in eps])),
        ]:
            val1 = metric_fn(v1e)
            val2 = metric_fn(v2e)
            delta = val2 - val1
            if metric_name == "MeanLen":
                print(f"{step:>5} | {metric_name:>12} | {val1:>8.0f} | {val2:>8.0f} | {delta:>+8.0f}")
            else:
                print(f"{step:>5} | {metric_name:>12} | {val1:>8.3f} | {val2:>8.3f} | {delta:>+8.3f}")
        print("-" * 55)

    # =========================================================================
    # 3. Same-question comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Same-Question Comparison")
    print("=" * 80)

    # Group episodes by (step, question_id)
    def group_by_sq(episodes):
        groups = defaultdict(list)
        for ep in episodes:
            groups[(ep["step"], ep["question_id"])].append(ep)
        return groups

    v1_sq = group_by_sq(v1_episodes)
    v2_sq = group_by_sq(v2_episodes)

    matched_keys = sorted(set(v1_sq.keys()) & set(v2_sq.keys()))
    print(f"\nMatched (step, question) pairs: {len(matched_keys)}")

    print(f"\n{'Step':>4} | {'Q (first 50 chars)':50} | {'v1_N':>4} | {'v2_N':>4} | {'v1_Acc':>6} | {'v2_Acc':>6} | {'v1_Trunc':>8} | {'v2_Trunc':>8} | {'v1_Len':>7} | {'v2_Len':>7}")
    print("-" * 170)

    comparison_data = []
    for key in matched_keys:
        step, qid = key
        v1_eps = v1_sq[key]
        v2_eps = v2_sq[key]
        v1_acc = np.mean([e["correct"] for e in v1_eps])
        v2_acc = np.mean([e["correct"] for e in v2_eps])
        v1_trunc = np.mean([not e["eos"] for e in v1_eps])
        v2_trunc = np.mean([not e["eos"] for e in v2_eps])
        v1_len = np.mean([e["response_len"] for e in v1_eps])
        v2_len = np.mean([e["response_len"] for e in v2_eps])
        v1_n_correct = sum(e["correct"] for e in v1_eps)
        v2_n_correct = sum(e["correct"] for e in v2_eps)
        v1_n_trunc = sum(not e["eos"] for e in v1_eps)
        v2_n_trunc = sum(not e["eos"] for e in v2_eps)

        print(f"{step:>4} | {qid[:50]:50} | {len(v1_eps):>4} | {len(v2_eps):>4} | {v1_acc:>6.3f} | {v2_acc:>6.3f} | {v1_n_trunc:>3}/{len(v1_eps):<4} | {v2_n_trunc:>3}/{len(v2_eps):<4} | {v1_len:>7.0f} | {v2_len:>7.0f}")

        comparison_data.append({
            "step": step,
            "question_id": qid,
            "v1_acc": v1_acc,
            "v2_acc": v2_acc,
            "v1_n_correct": v1_n_correct,
            "v2_n_correct": v2_n_correct,
            "v1_n": len(v1_eps),
            "v2_n": len(v2_eps),
            "v1_trunc": v1_trunc,
            "v2_trunc": v2_trunc,
            "v1_len": v1_len,
            "v2_len": v2_len,
        })

    # Aggregate over matched questions
    print(f"\n--- Aggregate over matched questions ---")
    v1_accs = [d["v1_acc"] for d in comparison_data]
    v2_accs = [d["v2_acc"] for d in comparison_data]
    v1_lens = [d["v1_len"] for d in comparison_data]
    v2_lens = [d["v2_len"] for d in comparison_data]
    v1_truncs = [d["v1_trunc"] for d in comparison_data]
    v2_truncs = [d["v2_trunc"] for d in comparison_data]

    print(f"Mean accuracy:  v1={np.mean(v1_accs):.3f}, v2={np.mean(v2_accs):.3f}")
    print(f"Mean length:    v1={np.mean(v1_lens):.0f}, v2={np.mean(v2_lens):.0f}")
    print(f"Mean trunc%:    v1={np.mean(v1_truncs):.3f}, v2={np.mean(v2_truncs):.3f}")

    # =========================================================================
    # 4. Qualitative diff — first 5 matched questions
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Qualitative Diff (first 5 matched questions)")
    print("=" * 80)

    for idx, key in enumerate(matched_keys[:5]):
        step, qid = key
        v1_eps = v1_sq[key]
        v2_eps = v2_sq[key]

        print(f"\n--- Match {idx+1}: Step {step} ---")
        print(f"Question: {qid[:80]}...")
        print(f"Ref answer: {v1_eps[0]['ref_answer']}")
        print(f"v1: {len(v1_eps)} rollouts, {sum(e['correct'] for e in v1_eps)}/{len(v1_eps)} correct, mean len={np.mean([e['response_len'] for e in v1_eps]):.0f}")
        print(f"v2: {len(v2_eps)} rollouts, {sum(e['correct'] for e in v2_eps)}/{len(v2_eps)} correct, mean len={np.mean([e['response_len'] for e in v2_eps]):.0f}")

        # Show first rollout from each
        print(f"\nv1 rollout[0] (first 300 chars):")
        print(f"  {v1_eps[0]['response'][:300]}...")
        print(f"\nv2 rollout[0] (first 300 chars):")
        print(f"  {v2_eps[0]['response'][:300]}...")

        # Structural comparison
        v1_think_lens = []
        v2_think_lens = []
        for e in v1_eps:
            think_match = re.search(r"<think>(.*?)</think>", e["response"], re.DOTALL)
            if think_match:
                v1_think_lens.append(len(think_match.group(1)))
            else:
                v1_think_lens.append(0)
        for e in v2_eps:
            think_match = re.search(r"<think>(.*?)</think>", e["response"], re.DOTALL)
            if think_match:
                v2_think_lens.append(len(think_match.group(1)))
            else:
                v2_think_lens.append(0)

        print(f"\nThink block lengths: v1 mean={np.mean(v1_think_lens):.0f}, v2 mean={np.mean(v2_think_lens):.0f}")

    # =========================================================================
    # 5. Statistical test
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: Statistical Test (Paired Sign Test)")
    print("=" * 80)

    wins = 0  # v2 > v1
    losses = 0  # v2 < v1
    ties = 0  # v2 == v1

    for d in comparison_data:
        if d["v2_acc"] > d["v1_acc"]:
            wins += 1
        elif d["v2_acc"] < d["v1_acc"]:
            losses += 1
        else:
            ties += 1

    print(f"\nFor each (question, step) pair, is v2 accuracy >= v1?")
    print(f"  v2 wins:  {wins}")
    print(f"  v2 losses: {losses}")
    print(f"  Ties:     {ties}")
    print(f"  Total:    {wins + losses + ties}")

    # Sign test (exclude ties)
    n_nontied = wins + losses
    if n_nontied > 0:
        # Two-sided binomial test: under H0 p(win) = 0.5
        p_value = stats.binomtest(wins, n_nontied, 0.5).pvalue
        print(f"\n  Binomial sign test (excluding ties): p = {p_value:.4f}")
        print(f"  (H0: P(v2 > v1) = 0.5 among non-tied pairs)")
    else:
        print("\n  All pairs tied — no test possible.")

    # Also do a paired t-test on accuracies for context
    if len(comparison_data) > 1:
        diffs = [d["v2_acc"] - d["v1_acc"] for d in comparison_data]
        t_stat, p_ttest = stats.ttest_rel(v2_accs, v1_accs)
        print(f"\n  Paired t-test on per-question accuracy: t={t_stat:.3f}, p={p_ttest:.4f}")
        print(f"  Mean diff (v2 - v1): {np.mean(diffs):.4f} ± {np.std(diffs, ddof=1):.4f}")

    # Length comparison test
    if len(comparison_data) > 1:
        len_diffs = [d["v2_len"] - d["v1_len"] for d in comparison_data]
        t_len, p_len = stats.ttest_rel(v2_lens, v1_lens)
        print(f"\n  Paired t-test on per-question mean length: t={t_len:.3f}, p={p_len:.4f}")
        print(f"  Mean len diff (v2 - v1): {np.mean(len_diffs):.0f} ± {np.std(len_diffs, ddof=1):.0f}")

    # =========================================================================
    # 6. v1-only trajectory over all 20 steps
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: v1 Full Trajectory (20 steps) for context")
    print("=" * 80)
    print_step_table("v1 full trajectory", v1_by_step)


if __name__ == "__main__":
    main()
