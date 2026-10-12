"""Freeze training/evaluation rows, remove exact problem overlap, and save provenance."""
import argparse
import hashlib
import json
import re
import random
from pathlib import Path

from common import extract_answer, file_hash, write_json


def normalize(problem):
    return re.sub(r"\s+", "", problem).lower()


def problem_from_dapo(row):
    content = "\n".join(m["content"] for m in row["prompt"] if m["role"] == "user")
    # Strip the known DAPO answer-format wrapper and impose our common boxed prompt.
    if content.startswith("Solve the following math problem step by step.") and "\n\n" in content:
        content = content.split("\n\n", 1)[1]
    content = re.split(r"\n\nRemember to put your answer", content, maxsplit=1)[0]
    return content.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--max-scan", type=int, default=1000000)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    train_id = "EleutherAI/hendrycks_math"
    eval_id = "HuggingFaceH4/MATH-500"
    api = HfApi()
    print("[DATA] Resolving dataset revisions and downloading evaluation data", flush=True)
    train_revision = api.dataset_info(train_id).sha
    eval_revision = api.dataset_info(eval_id).sha
    evaluation = load_dataset(eval_id, split="test", revision=eval_revision)
    eval_rows = [{"id": str(i), "problem": r["problem"], "answer": r["answer"]}
                 for i, r in enumerate(evaluation)]
    eval_keys = {normalize(r["problem"]) for r in eval_rows}
    if not 1 <= args.eval_size <= len(eval_rows):
        raise ValueError("eval-size must be between 1 and 500")
    # Remove exact overlaps with all 500 questions, then freeze a random evaluation subset.
    eval_rows = sorted(random.Random(args.seed).sample(eval_rows, args.eval_size), key=lambda r: int(r["id"]))
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    # Read train-only Parquet shards pinned to the dataset commit; no dataset scripts.
    files = [f"hf://datasets/{train_id}@{train_revision}/{subject}/train-00000-of-00001.parquet"
             for subject in subjects]
    train = load_dataset("parquet", data_files={"train": files}, split="train")
    seen, pool = set(), []
    overlaps = duplicates = scanned = missing_gold = outside_level = 0
    for row in train:
        scanned += 1
        level_match = re.fullmatch(r"Level ([1-5])", str(row["level"]).strip())
        if not level_match or int(level_match.group(1)) not in (1, 2, 3):
            outside_level += 1
            continue
        answer = extract_answer(row["solution"])
        if answer is None:
            missing_gold += 1
            continue
        problem = row["problem"].strip()
        key = normalize(problem)
        if key in eval_keys:
            overlaps += 1
        elif key in seen:
            duplicates += 1
        else:
            seen.add(key)
            pool.append({"id": hashlib.sha256(key.encode()).hexdigest()[:16],
                         "problem": problem, "answer": answer,
                         "level": int(level_match.group(1)), "subject": row["type"]})
    if len(pool) < args.train_size:
        raise RuntimeError(f"Only {len(pool)} eligible MATH Level 1-3 train rows")
    selected = random.Random(args.seed).sample(pool, args.train_size)
    print(f"[DATA] MATH train Level 1-3: eligible={len(pool)} selected={len(selected)}", flush=True)
    for name, rows in [("train", selected), ("eval", eval_rows)]:
        with (root / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(root / "manifest.json", {
        "train_dataset": train_id, "train_revision": train_revision,
        "eval_dataset": eval_id, "eval_revision": eval_revision,
        "scanned": scanned, "train_count": len(selected), "eval_count": len(eval_rows),
        "eval_subset_seed": args.seed,
        "train_split": "train", "train_levels": [1, 2, 3],
        "eligible_train_count": len(pool), "missing_gold": missing_gold,
        "outside_level": outside_level, "train_subset_seed": args.seed,
        "removed_exact_overlap": overlaps, "removed_duplicates": duplicates,
        "decontamination": "Whitespace-normalized exact match only; paraphrases not detected.",
        "train_sha256": file_hash(root / "train.jsonl"), "eval_sha256": file_hash(root / "eval.jsonl")})
    print(f"Saved {len(selected)} train rows and {len(eval_rows)} evaluation rows to {root}")


if __name__ == "__main__":
    main()
