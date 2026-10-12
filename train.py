"""Single-GPU DAPO + LoRA. Batched same-prompt rollouts; per-response backpropagation."""
import argparse
import json
import random
import time
from pathlib import Path

from common import (completion_logps, file_hash, generate_group, grade, extract_answer, load_model,
                    prompt_for, read_jsonl, seed_all, write_json)
from dapo_math import group_advantages, loss_sum, mixed_correctness, soft_overlong_penalty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--out", default="runs/dapo")
    p.add_argument("--resume", help="Checkpoint directory, including trainer.pt")
    p.add_argument("--max-seconds", type=float, help="Stop at a safe boundary and save the last completed update")
    args = p.parse_args()
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else float("inf")
    import numpy as np
    import torch
    cfg = json.loads(Path(args.config).read_text())
    if cfg["num_generations"] < 2 or cfg["groups_per_step"] < 1 or cfg["update_epochs"] < 1:
        raise ValueError("Need >=2 generations, >=1 group and >=1 update epoch")
    soft_overlong_penalty(0, cfg["max_completion_tokens"], cfg["overlong_buffer"])
    if args.resume:
        saved = json.loads((Path(args.resume) / "run_config.json").read_text())
        # Only the total requested steps may change on resume.
        requested_steps = cfg["steps"]
        for key in cfg:
            if key not in {"steps", "revision"} and cfg[key] != saved[key]:
                raise ValueError(f"Resume configuration changed: {key}")
        cfg = saved
        cfg["steps"] = requested_steps
        if cfg["train_sha256"] != file_hash(args.data):
            raise ValueError("Training data changed since checkpoint")
    seed_all(cfg["seed"])
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Empty training data")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Loading {cfg['model']} for training", flush=True)
    model, tokenizer = load_model(cfg, adapter=args.resume, train=True)
    print("[MODEL] Loaded. Starting DAPO sampling.", flush=True)
    cfg["train_sha256"] = file_hash(args.data)
    write_json(root / "run_config.json", cfg)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=cfg["learning_rate"], weight_decay=0.0)
    start_step = 0
    if args.resume:
        # Load only your own checkpoint; trainer.pt contains Python RNG state.
        state = torch.load(Path(args.resume) / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    def save(step):
        target = root / f"step-{step:05d}"
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        write_json(target / "run_config.json", cfg)
        torch.save({"step": step, "optimizer": optimizer.state_dict(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                   target / "trainer.pt")
        print(f"Checkpoint: {target}", flush=True)
        write_json(root / "latest_checkpoint.json", {"path": str(target), "step": step})

    for step in range(start_step + 1, cfg["steps"] + 1):
        if time.monotonic() >= deadline:
            save(step - 1)
            print("Training time budget reached; saved completed updates.", flush=True)
            return
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        batch, accepted = [], 0
        candidates = all_correct = all_wrong = too_long_prompt = truncations = 0
        correctness_sum = generated_count = 0
        sampling_seconds = 0.0
        generated_tokens = 0
        reward_sum = 0.0
        missing_answers = 0
        while accepted < cfg["groups_per_step"]:
            if time.monotonic() >= deadline:
                save(step - 1)
                print("Training time budget reached while sampling.", flush=True)
                return
            candidates += 1
            print(f"[ROLLOUT] step={step}/{cfg['steps']} candidate={candidates} accepted={accepted}/{cfg['groups_per_step']}", flush=True)
            if candidates > cfg["max_candidate_groups"]:
                save(step - 1)
                raise RuntimeError("Dynamic sampling could not fill a batch. See README: all-zero groups.")
            row = random.choice(rows)
            prompt = prompt_for(row["problem"])
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > cfg["max_prompt_tokens"]:
                too_long_prompt += 1
                continue
            group, correct, rewards = [], [], []
            print("\n[QUESTION]\n" + row["problem"], flush=True)
            print("[GOLD ANSWER] " + str(row["answer"]), flush=True)
            sampling_started = time.monotonic()
            print(f"[BATCH] Generating {cfg['num_generations']} replies together", flush=True)
            replies = generate_group(model, tokenizer, prompt, cfg["max_completion_tokens"],
                                     cfg["num_generations"])
            sampling_seconds += time.monotonic() - sampling_started
            for ids, n_prompt, text, truncated in replies:
                ok = grade(text, row["answer"])
                length = len(ids) - n_prompt
                reward = (1.0 if ok else -1.0) + soft_overlong_penalty(
                    length, cfg["max_completion_tokens"], cfg["overlong_buffer"])
                generated_tokens += length
                reward_sum += reward
                missing_answers += int(extract_answer(text) is None)
                # Save every candidate, including rejected groups, for grader/error inspection.
                with (root / "rollout_audit.jsonl").open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps({"step": step, "candidate": candidates,
                        "answer_index": len(group), "problem": row["problem"],
                        "gold": row["answer"], "response": text,
                        "extracted_answer": extract_answer(text), "correct": ok,
                        "truncated": truncated, "tokens": length, "reward": reward}) + "\n")
                print(f"\n[RESPONSE {len(group)+1}/{cfg['num_generations']}]\n{text}", flush=True)
                print(f"[EXTRACTED ANSWER] {extract_answer(text)!r} | correct={ok} | truncated={truncated}", flush=True)
                correct.append(ok)
                rewards.append(reward)
                group.append({"ids": ids.cpu(), "n_prompt": n_prompt, "truncated": truncated})
                truncations += int(truncated)
                print(f"[SAMPLE] answer={len(group)}/{cfg['num_generations']} tokens={length} correct={ok} truncated={truncated}", flush=True)
            correctness_sum += sum(correct)
            generated_count += len(correct)
            # DAPO dynamic sampling filters on correctness, not on length-shaped reward.
            if not mixed_correctness(correct):
                print("[GROUP] Rejected: all answers have the same correctness", flush=True)
                all_correct += int(all(correct))
                all_wrong += int(not any(correct))
                continue
            if all(g["truncated"] for g in group):
                print("[GROUP] Rejected: all responses truncated", flush=True)
                continue
            advantages = group_advantages(rewards)
            model.eval()
            for item, advantage in zip(group, advantages):
                # Mask hard-truncated responses from policy loss (overlong filtering).
                if item["truncated"]:
                    continue
                ids = item["ids"].to(model.device)
                with torch.no_grad():
                    item["old_logps"] = completion_logps(model, ids, item["n_prompt"]).cpu()
                item["advantage"] = float(advantage)
                batch.append(item)
            accepted += 1
            print(f"[GROUP] Accepted: {sum(correct)}/{len(correct)} correct; collected={accepted}/{cfg['groups_per_step']}", flush=True)
        # DAPO token-level aggregation: ONE denominator over every valid response token.
        denominator = sum(len(item["old_logps"]) for item in batch)
        epoch_losses = []
        gradient_norms = []
        ratio_deviations = []
        for _ in range(cfg["update_epochs"]):
            print(f"[UPDATE] step={step} epoch={_+1}/{cfg['update_epochs']} tokens={denominator}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total_loss = 0.0
            ratio_deviation_sum = 0.0
            for item in batch:
                logps = completion_logps(model, item["ids"].to(model.device), item["n_prompt"])
                with torch.no_grad():
                    ratio_deviation_sum += (torch.exp(logps - item["old_logps"].to(model.device)) - 1).abs().sum().item()
                loss = loss_sum(logps, item["old_logps"].to(model.device), item["advantage"],
                                cfg["clip_low"], cfg["clip_high"]) / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite policy loss; stop instead of saving corrupted weights")
                loss.backward()
                total_loss += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            gradient_norms.append(float(norm))
            ratio_deviations.append(ratio_deviation_sum / denominator)
            optimizer.step()
            epoch_losses.append(total_loss)
        record = {"step": step, "policy_loss": epoch_losses,
                  "gradient_norms": gradient_norms,
                  "mean_abs_ratio_change": ratio_deviations,
                  "acceptance_rate": accepted / candidates,
                  "mean_reward": reward_sum / max(1, generated_count),
                  "missing_answer_rate": missing_answers / max(1, generated_count),
                  "sampling_seconds": sampling_seconds,
                  "generated_tokens": generated_tokens,
                  "sampling_tokens_per_second": generated_tokens / max(1e-6, sampling_seconds),
                  "candidate_groups": candidates, "accepted_groups": accepted,
                  "all_correct_groups": all_correct, "all_wrong_groups": all_wrong,
                  "too_long_prompts": too_long_prompt,
                  "candidate_accuracy": correctness_sum / max(1, generated_count),
                  "truncation_rate": truncations / max(1, generated_count),
                  "valid_tokens": denominator, "seconds": time.time() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        with (root / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            save(step)


if __name__ == "__main__":
    main()
