"""Paired accuracy delta and paired bootstrap interval; refuses mismatched protocols."""
import argparse
import json
from pathlib import Path

from common import read_jsonl, write_json


def summarize(base, trained, seed=42):
    import numpy as np
    b = np.asarray(base, dtype=float)
    t = np.asarray(trained, dtype=float)
    if len(b) == 0 or len(b) != len(t):
        raise ValueError("Need nonempty paired results")
    difference = t - b
    rng = np.random.default_rng(seed)
    samples = [difference[rng.integers(0, len(b), len(b))].mean() for _ in range(10000)]
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"n": len(b), "base_accuracy": float(b.mean()), "dapo_accuracy": float(t.mean()),
            "absolute_gain_pp": float(difference.mean() * 100),
            "relative_gain_percent": float((t.mean()/b.mean()-1)*100) if b.mean() else None,
            "paired_bootstrap_95ci_pp": [float(low*100), float(high*100)],
            "improved_questions": int((t>b).sum()), "regressed_questions": int((t<b).sum()),
            "unchanged_questions": int((t==b).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="results/base")
    p.add_argument("--trained", default="results/dapo")
    p.add_argument("--out", default="results/comparison")
    args = p.parse_args()
    base_root, trained_root = Path(args.base), Path(args.trained)
    bm = json.loads((base_root / "metadata.json").read_text())
    tm = json.loads((trained_root / "metadata.json").read_text())
    if bm["adapter"] is not None or tm["adapter"] is None:
        raise ValueError("Expected base without adapter and trained with adapter")
    if bm["protocol"] != tm["protocol"]:
        raise ValueError("Different evaluation protocols. Use the checkpoint run_config.json for both runs.")
    def indexed(root):
        rows = read_jsonl(root / "predictions.jsonl")
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate evaluation IDs")
        return result
    b, t = indexed(base_root), indexed(trained_root)
    ids = bm["protocol"]["ids"]
    if set(b) != set(ids) or set(t) != set(ids):
        raise ValueError("Incomplete evaluation; resume both before comparison")
    for key in ids:
        if b[key]["problem"] != t[key]["problem"] or b[key]["gold"] != t[key]["gold"]:
            raise ValueError("Problem/gold mismatch")
    result = summarize([b[k]["accuracy"] for k in ids], [t[k]["accuracy"] for k in ids])
    root = Path(args.out)
    write_json(root / "comparison.json", result)
    ci = result["paired_bootstrap_95ci_pp"]
    report = (f"# Base vs DAPO\n\n"
              f"Questions: {result['n']}\n\n"
              f"| Model | Accuracy |\n|---|---:|\n"
              f"| Base | {result['base_accuracy']:.2%} |\n"
              f"| DAPO + LoRA | {result['dapo_accuracy']:.2%} |\n\n"
              f"Absolute change: **{result['absolute_gain_pp']:+.2f} percentage points**.\n\n"
              f"Paired bootstrap 95% interval: [{ci[0]:+.2f}, {ci[1]:+.2f}] percentage points.\n\n"
              f"Improved: {result['improved_questions']}; regressed: {result['regressed_questions']}; "
              f"unchanged: {result['unchanged_questions']}.\n\n"
              "An interval crossing zero is inconclusive. This interval captures question-sampling "
              "uncertainty, not variability across independent training seeds. No gain is guaranteed.\n")
    (root / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
