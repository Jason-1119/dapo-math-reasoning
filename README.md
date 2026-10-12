# DAPO Math Reasoning

A single-GPU experiment in reinforcement learning for mathematical reasoning: train **Qwen3-1.7B-Base** with **DAPO + LoRA**, then compare its sampled answer accuracy against the original base model on held-out **MATH500** questions.

The project includes batched rollouts, inspectable rewards and answers, an isolated Google Colab environment, and paired evaluation. 

## Reported experiment

One completed Colab run produced the following results on a fixed 50-question MATH500 subset:

| Model | Correct answers | Accuracy |
|---|---:|---:|
| Qwen3-1.7B-Base | 13 / 50 | 26.0% |
| DAPO + LoRA | 17 / 50 | 34.0% |

- **Observed difference:** +8.0 percentage points.
- **Question-level changes:** 6 improved, 2 regressed, 42 unchanged.
- **Completed rollout steps:** 20.

The values above were transcribed from the completed run report. For an auditable reproduction, retain that run's `run_config.json`, data manifest, evaluation metadata, predictions, and comparison report; the current default configuration alone does not establish a historical run's settings. These scores should not be compared directly with paper results obtained using different training and evaluation protocols.

## Method

DAPO builds on group-relative policy optimization. For each candidate training question, the model samples four responses together. Groups containing both correct and incorrect answers are retained; all-correct and all-incorrect groups are rejected.

The implementation includes:

- **Clip-Higher:** asymmetric policy-ratio clipping with `clip_low=0.20` and `clip_high=0.28`.
- **Dynamic sampling:** collect the requested number of accepted groups before updating.
- **Group-relative advantages:** normalize correctness and length-shaped rewards within each group.
- **Token-level policy loss:** normalize by the total number of valid response tokens across accepted groups.
- **Overlong handling:** apply a soft length penalty and exclude hard-truncated responses from policy loss.
- **LoRA updates:** optimize adapters while keeping original base weights frozen.

Correctness rewards are +1 for a correct answer and −1 for an incorrect answer, with an additional length penalty. Training samples from temperature 1.0, top-p 1.0, and top-k 0 to match the distribution used for policy scoring. Old-policy log probabilities are fixed across the two update epochs. No learned critic or reward model is required.

Each step collects four accepted groups, then performs two optimizer updates through gradient accumulation. Twenty steps therefore correspond to 40 optimizer updates and 80 accepted groups, which may include repeated questions. Candidate groups rejected during sampling do not contribute to the policy loss. Responses are generated in batches within a question; candidate questions and response backpropagation remain sequential.

## Data

### Training

- Source: [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
- Split: **train only**, across all seven subjects.
- Difficulty: **Level 1–3**.
- Selection: 512 unique eligible questions sampled with seed 42.
- Gold answers: extracted from the final balanced `\boxed{...}` in the provided solution; rows without extractable answers are skipped.

The solution and gold answer are used for grading and inspection, not supplied to the model's generation prompt.

### Evaluation

- Source: [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
- Final comparison: a seeded 50-question subset, identical for both models.
- Preliminary check: 10 fixed questions from that subset, evaluated before training.

Data preparation excludes whitespace-normalized exact training-question matches against **all 500** MATH500 questions. Dataset revisions and JSONL hashes are recorded. This check does not detect semantic duplicates, paraphrases, or pretraining contamination.

The 10-question check is a quick diagnostic, not training-set accuracy or a reliable benchmark estimate. It must not be used to select training questions, checkpoints, or hyperparameters. Use a separate validation split for tuning.

## Default configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Training questions | 512 MATH train Level 1–3 |
| Rollout steps | 20 |
| Accepted groups per step | 4 |
| Responses per candidate question | 4, generated together |
| Maximum generated response length | 2,048 tokens for training and evaluation |
| Maximum prompt length | 768 tokens; longer training prompts are skipped |
| Overlong penalty buffer | 512 tokens, within the response limit |
| Learning rate | `5e-6` |
| Update epochs per rollout step | 2 |
| LoRA rank / alpha | 16 / 32 |
| Gradient clipping | 1.0 |
| Maximum candidate groups per step | 128 |
| Final evaluation questions | 50 |
| Seed | 42 |

Edit `config.json` to change training parameters. The candidate-group cap reports a sampling failure if an effective batch cannot be filled; it is not a time limit. **The default notebook sets no setup, training, or experiment deadline.** Optional CLI time-limit flags apply only when explicitly supplied.

Training duration depends on answer length, hardware, and the fraction of accepted groups. Batching improves the generation structure but does not guarantee a particular runtime or accuracy gain.

## Run in Google Colab

1. Download this repository and upload its contents to Google Drive in a folder named `dapo-base-compare`:

   ```text
   MyDrive/dapo-base-compare/
   ```

   Keep the Python scripts, `config.json`, `requirements.txt`, and `tests/` together with `colab.ipynb`. If using a different folder name, edit `SOURCE` in the notebook's first cell.

2. Open **`colab.ipynb`** in Colab and select a BF16-capable CUDA GPU. 
3. Run the mount/copy cell, followed by the environment setup and preflight cell.
4. Run the **10-question base check** cell to inspect initial accuracy and truncation.
5. Run the training/comparison cell. It prepares data, trains adapters, evaluates the original base and final adapter checkpoint, and generates the comparison report.
6. Run the final display/save cell to show the report and copy run artifacts to Drive.


## Repository structure

```text
colab.ipynb           Colab setup, preliminary check, training, and results
config.json           Default experiment settings
requirements.txt      Pinned dependencies
setup_colab.py        Isolated environment installation
preflight.py          Dependency, CUDA, and smoke checks
prepare.py            MATH Level 1–3 selection and held-out evaluation data
train.py              DAPO + LoRA sampling and optimization
common.py             Prompts, grading, generation, and token log probabilities
dapo_math.py          Advantages, clipping objective, and length penalty
evaluate.py           Base or adapter evaluation
compare.py            Paired comparison and uncertainty estimates
run_budget.py         Experiment orchestration; optional time limits
live_process.py       Streaming subprocess output and logs
tests/                Implementation checks
```

`config_full.json`, if retained, is an alternative configuration rather than the default used by the notebook. `colab.ipynb` is the documented entry point.

## References

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Official DAPO implementation](https://github.com/BytedTsinghua-SIA/DAPO)
- [Learning Structured Reasoning via Tractable Trajectory Control](https://arxiv.org/abs/2603.01641)
- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [MATH training dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [MATH500 evaluation dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
