"""Run training and comparison to completion; time limits are opt-in."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from live_process import run_live


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-minutes", type=float, default=None)
    p.add_argument("--train-minutes", type=float, default=None)
    p.add_argument("--eval-size", type=int, default=50)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--config", default="config.json")
    args = p.parse_args()
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60 if args.budget_minutes else float("inf")
    if any(value is not None and value <= 0 for value in (args.budget_minutes, args.train_minutes)):
        raise ValueError("Time budgets must be positive")
    run_id = "budget-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    train = root / "dapo"
    results = Path("results") / run_id
    config_path = root / "config.json"
    cfg = json.loads(Path(args.config).read_text())
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    env = dict(os.environ, HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="30")

    def run(*command, seconds=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, args.budget_minutes * 60)
        stage = str(command[0])
        status.update(stage=stage, elapsed_minutes=(time.monotonic()-started)/60)
        (root/"budget_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        limit_label = f"{remaining/60:.1f} min" if args.budget_minutes else "none"
        print(f"\n[STAGE] {stage} | elapsed={status['elapsed_minutes']:.1f} min | time_limit={limit_label}", flush=True)
        run_live([sys.executable, "-u", *map(str, command)], env=env,
                 timeout=min(remaining, seconds) if seconds else (remaining if args.budget_minutes else None),
                 log_path=root/"console.log", label=stage, new_session=False)

    status = {"run": run_id, "budget_minutes": args.budget_minutes,
              "eval_size": args.eval_size, "status": "running"}
    try:
        run("prepare.py", "--out", data, "--train-size", args.train_size,
            "--eval-size", args.eval_size, "--seed", cfg["seed"])
        train_command = ["train.py", "--config", config_path, "--data", data/"train.jsonl", "--out", train]
        if args.train_minutes:
            train_command.extend(["--max-seconds", args.train_minutes*60])
        run(*train_command)
        latest = json.loads((train/"latest_checkpoint.json").read_text())
        if latest["step"] == 0:
            raise RuntimeError("No training update completed. Do not interpret this as a DAPO result.")
        checkpoint = Path(latest["path"])
        frozen_config = checkpoint/"run_config.json"
        run("evaluate.py", "--config", frozen_config, "--data", data/"eval.jsonl",
            "--out", results/"base")
        run("evaluate.py", "--config", frozen_config, "--data", data/"eval.jsonl",
            "--adapter", checkpoint, "--out", results/"dapo")
        run("compare.py", "--base", results/"base", "--trained", results/"dapo",
            "--out", results/"comparison")
        status.update(status="complete", completed_training_steps=latest["step"],
                      report=str(results/"comparison/report.md"))
    except subprocess.TimeoutExpired:
        status.update(status="time_limit", message="Budget exhausted. Partial outputs are preserved; no complete comparison is claimed.")
    except (subprocess.CalledProcessError, RuntimeError) as error:
        status.update(status="failed", message=str(error))
    finally:
        status["elapsed_minutes"] = (time.monotonic()-started)/60
        (root/"budget_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
    if status["status"] != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
