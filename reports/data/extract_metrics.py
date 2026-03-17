#!/usr/bin/env python3
"""Extract and consolidate all RLVR metrics into a single JSON file.

Reads metrics.jsonl from all 3 v3b runs and outputs reports/data/all_metrics.json.
"""

import json
from pathlib import Path

RUNS = {
    "gptoss-20b": "/tmp/tinker-examples/rlvr/gptoss-20b-v3b/metrics.jsonl",
    "qwen3-30b-instruct": "/tmp/tinker-examples/rlvr/qwen3-30b-instruct-v3b/metrics.jsonl",
    "qwen3-30b-think": "/tmp/tinker-examples/rlvr/qwen3-30b-think-v3b/metrics.jsonl",
}

# Fields to extract for train rows
TRAIN_FIELDS = {
    "correct": "env/all/correct",
    "format_boxed": "env/all/format_boxed",
    "format_eos": "env/all/format_eos",
    "ac_tokens_per_turn": "env/all/ac_tokens_per_turn",
    "reward": "env/all/reward/total",
    "total_episodes": "env/all/total_episodes",
    "frac_mixed": "env/all/by_group/frac_mixed",
    "frac_all_good": "env/all/by_group/frac_all_good",
    "frac_all_bad": "env/all/by_group/frac_all_bad",
}

# Additional train-only fields (optimizer/timing)
TRAIN_EXTRA = {
    "kl": "optim/kl_sample_train_v2",
    "entropy": "optim/entropy",
    "lr": "optim/lr",
    "grad_norm": "unclipped_grad_l2:mean",
}

# Fields to extract for eval rows
EVAL_FIELDS = {
    "correct": "test/test/env/all/correct",
    "format_boxed": "test/test/env/all/format_boxed",
    "format_eos": "test/test/env/all/format_eos",
    "ac_tokens_per_turn": "test/test/env/all/ac_tokens_per_turn",
    "reward": "test/test/env/all/reward/total",
    "total_episodes": "test/test/env/all/total_episodes",
    "frac_all_good": "test/test/env/all/by_group/frac_all_good",
    "frac_all_bad": "test/test/env/all/by_group/frac_all_bad",
}


def is_eval_row(row: dict) -> bool:
    return "test/test/env/all/correct" in row


def is_train_row(row: dict) -> bool:
    return "env/all/correct" in row


def extract_row(run: str, raw: dict) -> dict | None:
    step = raw.get("step")
    if step is None:
        return None

    if is_eval_row(raw):
        out = {"run": run, "step": step, "type": "eval"}
        for name, key in EVAL_FIELDS.items():
            out[name] = raw.get(key)
        # Eval rows don't have kl/entropy/lr/grad_norm
        out["kl"] = None
        out["entropy"] = None
        out["lr"] = None
        out["grad_norm"] = None
        return out
    elif is_train_row(raw):
        out = {"run": run, "step": step, "type": "train"}
        for name, key in TRAIN_FIELDS.items():
            out[name] = raw.get(key)
        for name, key in TRAIN_EXTRA.items():
            out[name] = raw.get(key)
        return out
    else:
        return None


def main():
    all_rows = []
    stats = {}

    for run_name, path in RUNS.items():
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {path} not found, skipping")
            continue

        lines = p.read_text().strip().split("\n")
        train_count = 0
        eval_count = 0

        for line in lines:
            raw = json.loads(line)
            row = extract_row(run_name, raw)
            if row is not None:
                all_rows.append(row)
                if row["type"] == "train":
                    train_count += 1
                else:
                    eval_count += 1

        stats[run_name] = {
            "raw_lines": len(lines),
            "train": train_count,
            "eval": eval_count,
        }
        print(f"{run_name}: {len(lines)} raw lines -> {train_count} train + {eval_count} eval")

    # Sort by run, then type (eval before train at same step for readability), then step
    all_rows.sort(key=lambda r: (r["run"], r["step"], r["type"]))

    out_path = Path(__file__).parent / "all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "description": "Consolidated RLVR training metrics for 3 models on OmniMath",
                    "runs": list(RUNS.keys()),
                    "stats": stats,
                    "total_rows": len(all_rows),
                    "schema": {
                        "run": "model name",
                        "step": "training step",
                        "type": "train or eval",
                        "correct": "fraction correct",
                        "format_boxed": "fraction with \\boxed{} format",
                        "format_eos": "fraction with EOS format",
                        "ac_tokens_per_turn": "mean action tokens per turn",
                        "reward": "mean reward",
                        "total_episodes": "episodes in this batch",
                        "kl": "KL divergence (train only, v2)",
                        "entropy": "policy entropy (train only)",
                        "lr": "learning rate (train only)",
                        "grad_norm": "unclipped gradient L2 norm (train only)",
                        "frac_mixed": "fraction of groups with mixed outcomes (train only)",
                        "frac_all_good": "fraction of groups all correct",
                        "frac_all_bad": "fraction of groups all wrong",
                    },
                },
                "rows": all_rows,
            },
            f,
            indent=2,
        )

    print(f"\nWrote {len(all_rows)} rows to {out_path}")
    print(f"File size: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
