"""Shared prompts, strict answer grading, model loading, and token log probabilities."""
import hashlib
import json
import random
import re
from pathlib import Path

PROMPT_VERSION = "plain-boxed-v1"


def prompt_for(problem):
    # A base model gets a plain-text completion prompt, not an instruction chat template.
    return ("Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.\n\n" + problem.strip() + "\n\nSolution:\n")


def extract_answer(text):
    """Last balanced boxed answer, or explicit final Answer: line; never guess last number."""
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth, start = 1, match.end()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
    answers = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+)$", text)
    return answers[-1].strip() if answers else None


def grade(text, gold):
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig
    answer = extract_answer(text)
    if answer is None:
        return False
    try:
        # Gold and prediction use the same extraction format. Parse failures score zero.
        cfg = [LatexExtractionConfig()]
        target = parse("$" + str(gold).strip().strip("$") + "$", extraction_config=cfg)
        predicted = parse("$" + answer.strip().strip("$") + "$", extraction_config=cfg)
        return bool(target and predicted and verify(target, predicted))
    except Exception:
        return False


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config, adapter=None, train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("This project needs a CUDA GPU. Choose a GPU runtime in Colab.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if train and dtype != torch.bfloat16:
        raise RuntimeError("Training requires a BF16-capable GPU (G4/L4/A100). T4 is not supported by this preset.")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], torch_dtype=dtype,
        attn_implementation="sdpa", device_map={"": 0})
    # Record the resolved model commit so base and adapter evaluations load identical weights.
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved:
        config["revision"] = resolved
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if train:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.print_trainable_parameters()
    return model, tokenizer


def generate_one(model, tokenizer, prompt, max_tokens, temperature=1.0, top_p=1.0, top_k=0):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = inputs.input_ids.shape[1]
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=True, temperature=temperature,
                                top_p=top_p, top_k=top_k, max_new_tokens=max_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
    # Clone outside inference mode: these tensors may later be used by autograd.
    ids = output[0].clone()
    completion = ids[n_prompt:]
    truncated = len(completion) >= max_tokens and int(completion[-1]) != tokenizer.eos_token_id
    text = tokenizer.decode(completion, skip_special_tokens=True)
    return ids, n_prompt, text, truncated


def completion_length(token_ids, eos_token_id):
    """Exclude generation padding after the first EOS, retaining EOS for scoring."""
    for index, token in enumerate(token_ids):
        if int(token) == eos_token_id:
            return index + 1
    return len(token_ids)


def generate_group(model, tokenizer, prompt, max_tokens, num_generations):
    """Sample independent replies to ONE prompt in a single batched generate call."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = inputs.input_ids.shape[1]
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=True, temperature=1.0,
                                top_p=1.0, top_k=0, num_return_sequences=num_generations,
                                max_new_tokens=max_tokens, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
    replies = []
    for row in output:
        count = completion_length(row[n_prompt:].tolist(), tokenizer.eos_token_id)
        ids = row[:n_prompt + count].clone()
        completion = ids[n_prompt:]
        truncated = count >= max_tokens and int(completion[-1]) != tokenizer.eos_token_id
        replies.append((ids, n_prompt, tokenizer.decode(completion, skip_special_tokens=True), truncated))
    return replies


def completion_logps(model, ids, n_prompt):
    import torch
    # Only completion positions; chunk FP32 softmax to avoid a full [length,vocab] FP32 copy.
    logits = model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None],
                   use_cache=False).logits[0, n_prompt - 1:-1]
    labels = ids[n_prompt:]
    return torch.cat([-torch.nn.functional.cross_entropy(
        logits[i:i+128].float(), labels[i:i+128], reduction="none")
        for i in range(0, len(labels), 128)])
