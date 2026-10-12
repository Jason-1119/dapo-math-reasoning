"""Fixed MATH500 protocol; one sampled answer per problem by default (not pass@k)."""
import argparse
import json
from pathlib import Path

from common import (PROMPT_VERSION, extract_answer, file_hash, generate_one, grade,
                    load_model, prompt_for, read_jsonl, seed_all, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data/eval.jsonl")
    p.add_argument("--adapter", help="DAPO step directory; omit for base")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, help="Smoke test only; comparison requires matching limits")
    p.add_argument("--resume", action="store_true", help="Continue an interrupted evaluation")
    args = p.parse_args()
    import importlib.metadata
    cfg = json.loads(Path(args.config).read_text())
    if args.adapter:
        trained = json.loads((Path(args.adapter) / "run_config.json").read_text())
        if cfg["model"] != trained["model"]:
            raise ValueError("Evaluation base model does not match adapter")
        cfg["revision"] = trained["revision"]
    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"model": cfg["model"], "revision": cfg["revision"],
                "data_sha256": file_hash(args.data), "ids": [r["id"] for r in rows],
                "prompt_version": PROMPT_VERSION, "seed": cfg["seed"],
                "max_tokens": cfg["eval_max_tokens"], "samples": cfg["eval_samples"],
                "temperature": cfg["eval_temperature"], "top_p": cfg["eval_top_p"],
                "top_k": cfg["eval_top_k"], "metric": "mean sampled accuracy (not pass@k)",
                "software": {k: importlib.metadata.version(k) for k in
                             ["torch", "transformers", "peft", "math-verify"]}}
    seed_all(cfg["seed"])
    print(f"[MODEL] Loading {'DAPO adapter' if args.adapter else 'original base'} for {len(rows)} evaluation questions", flush=True)
    model, tokenizer = load_model(cfg, adapter=args.adapter)
    print("[EVAL] Model loaded. Starting generation.", flush=True)
    protocol["revision"] = cfg["revision"]
    records_path = root / "predictions.jsonl"
    done = {}
    if records_path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use a new directory or --resume")
        previous = json.loads((root / "metadata.json").read_text())
        if previous["protocol"] != protocol or previous["adapter"] != args.adapter:
            raise ValueError("Evaluation configuration changed; cannot resume")
        done = {r["id"]: r for r in read_jsonl(records_path)}
    write_json(root / "metadata.json", {"protocol": protocol, "adapter": args.adapter})
    for i, row in enumerate(rows):
        if row["id"] in done:
            continue
        predictions = []
        for sample in range(cfg["eval_samples"]):
            # Independent deterministic seed per row/sample allows interrupted evaluation resume.
            seed_all(cfg["seed"] + i * cfg["eval_samples"] + sample)
            _, _, text, truncated = generate_one(
                model, tokenizer, prompt_for(row["problem"]), cfg["eval_max_tokens"],
                cfg["eval_temperature"], cfg["eval_top_p"], cfg["eval_top_k"])
            predictions.append({"text": text, "answer": extract_answer(text),
                                "correct": grade(text, row["answer"]), "truncated": truncated})
        record = {"id": row["id"], "problem": row["problem"], "gold": row["answer"],
                  "accuracy": sum(v["correct"] for v in predictions) / len(predictions),
                  "predictions": predictions}
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        done[row["id"]] = record
        print(f"{i+1}/{len(rows)} accuracy={sum(v['accuracy'] for v in done.values())/len(done):.3f}", flush=True)
    write_json(root / "summary.json", {"n": len(done),
        "accuracy": sum(v["accuracy"] for v in done.values()) / len(done),
        "truncation_rate": sum(p["truncated"] for r in done.values() for p in r["predictions"]) /
                           (len(done) * cfg["eval_samples"])})


if __name__ == "__main__":
    main()
