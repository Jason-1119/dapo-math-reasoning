"""Fail before downloads/training if imports, CUDA, grading, or gradient tests fail."""
import importlib.metadata
import importlib.util
import json
import sys
import unittest
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedModel, Qwen3Config
    from peft import LoraConfig, PeftModel, get_peft_model
    from math_verify import parse, verify

    expected = {"torch": "2.8.0", "transformers": "4.56.2", "peft": "0.17.1",
                "datasets": "4.1.1", "accelerate": "1.10.1", "math-verify": "0.8.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        if versions[name].split("+")[0] != version:
            raise RuntimeError(f"Unexpected {name} version: {versions[name]}; expected {version}")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the isolated venv Python, not the Colab kernel interpreter")
    for name in ["torchvision", "torchaudio", "librosa"]:
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"Unexpected optional package {name} in the project venv")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select G4 or A100; a CUDA BF16-capable GPU is required")
    # A real CUDA forward/backward/generation test with a tiny random Qwen3, without downloading weights.
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                        pad_token_id=0, eos_token_id=2)
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                           task_type="CAUSAL_LM", lora_dropout=0.0))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.randint(3, 128, (1, 16), device="cuda")
    model.train()
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("CUDA gradient smoke test failed")
    model.eval()
    with torch.inference_mode():
        batched = model.generate(ids[:, :4], max_new_tokens=4, do_sample=True, num_return_sequences=4,
                       attention_mask=torch.ones_like(ids[:, :4]), use_cache=True)
    if batched.shape[0] != 4:
        raise RuntimeError("Batched generation smoke test failed")
    del model, ids, loss, grads, batched
    torch.cuda.empty_cache()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Tests failed or were skipped; do not start training")
    print(json.dumps({"python": sys.executable, "versions": versions,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                      "status": "imports, GPU forward/backward/generation, and all tests passed"}, indent=2))


if __name__ == "__main__":
    main()
