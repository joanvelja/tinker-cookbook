"""Monitor Q3-Instruct runs and kill each when it reaches step 5n+1 (checkpoint is safe)."""

import json
import os
import signal
import subprocess
import sys
import time

LOG_DIR = "logs/gpqa-q3instruct"
POLL_INTERVAL = 60  # seconds

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


def get_pid(run_name: str) -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"log_path={LOG_DIR}/{run_name}"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in result.stdout.strip().split() if p]
        # Return the python process (highest PID, the uv wrapper is lower)
        return max(pids) if pids else None
    except Exception:
        return None


def main():
    killed = set()
    print(f"Monitoring Q3-Instruct runs in {LOG_DIR}/")
    print(f"Will kill each run when it reaches step 5n+1 (n >= 1)")
    print(f"Polling every {POLL_INTERVAL}s. Ctrl+C to stop.\n")

    while True:
        all_done = True
        for entry in sorted(os.listdir(LOG_DIR)):
            run_dir = os.path.join(LOG_DIR, entry)
            if not os.path.isdir(run_dir) or entry in killed:
                continue

            step = get_latest_step(run_dir)
            pid = get_pid(entry)

            if pid is None:
                if entry not in killed:
                    print(f"  {entry}: no process (already dead or not started)")
                    killed.add(entry)
                continue

            all_done = False

            if step is None:
                print(f"  {entry}: PID {pid}, no metrics yet")
                continue

            # Kill at step 5n+1 for n >= 1 (i.e., step 6, 11, 16, 21, ...)
            if step >= 6 and (step % 5) >= 1:
                print(f"  {entry}: step {step} → KILLING (PID {pid}, checkpoint at step {(step // 5) * 5})")
                os.kill(pid, signal.SIGTERM)
                killed.add(entry)
            else:
                print(f"  {entry}: step {step}, PID {pid} — waiting")

        if all_done:
            print("\nAll Q3 runs killed or dead. Done.")
            break

        print(f"  [{len(killed)} killed, polling again in {POLL_INTERVAL}s...]\n")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
