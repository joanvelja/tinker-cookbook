"""Monitor ALL RLVR runs and kill each when it reaches step 5n+1 (checkpoint is safe).

Scans all log directories for running experiments. Polls every 60s.
Once all runs are killed, exits.

Usage:
    uv run python scripts/auto_shutdown_all.py
"""

import json
import os
import signal
import subprocess
import sys
import time

LOG_DIRS = [
    "logs/gpqa-experiment-v2",
    "logs/gpqa-experiment-v3",
    "logs/gpqa-q3instruct",
]
POLL_INTERVAL = 60


def get_latest_step(run_dir: str) -> int | None:
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None
    last_line = None
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        return None
    return json.loads(last_line).get("progress/batch", None)


def get_pid(log_dir: str, run_name: str) -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"log_path={log_dir}/{run_name}"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p]
        return max(pids) if pids else None
    except Exception:
        return None


def main():
    killed: set[str] = set()
    print("Auto-shutdown monitor for ALL RLVR runs")
    print(f"Directories: {LOG_DIRS}")
    print(f"Kill condition: step >= 6 and step % 5 >= 1 (checkpoint at 5n is safe)")
    print(f"Polling every {POLL_INTERVAL}s. Ctrl+C to stop.\n")
    sys.stdout.flush()

    while True:
        alive_count = 0

        for log_dir in LOG_DIRS:
            if not os.path.isdir(log_dir):
                continue
            for entry in sorted(os.listdir(log_dir)):
                run_dir = os.path.join(log_dir, entry)
                run_key = f"{log_dir}/{entry}"

                if not os.path.isdir(run_dir) or run_key in killed:
                    continue

                pid = get_pid(log_dir, entry)
                if pid is None:
                    if run_key not in killed:
                        killed.add(run_key)
                    continue

                step = get_latest_step(run_dir)
                if step is None:
                    print(f"  {run_key}: PID {pid}, no metrics yet")
                    alive_count += 1
                    continue

                if step >= 6 and (step % 5) >= 1:
                    ckpt = (step // 5) * 5
                    print(f"  {run_key}: step {step} → KILL (PID {pid}, ckpt at {ckpt})")
                    os.kill(pid, signal.SIGTERM)
                    killed.add(run_key)
                else:
                    print(f"  {run_key}: step {step}, PID {pid} — waiting")
                    alive_count += 1

        sys.stdout.flush()

        if alive_count == 0:
            print(f"\nAll runs killed or dead ({len(killed)} total). Safe to shut down.")
            break

        print(f"  [{alive_count} alive, {len(killed)} killed, next check in {POLL_INTERVAL}s]\n")
        sys.stdout.flush()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
