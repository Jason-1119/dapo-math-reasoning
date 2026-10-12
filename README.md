# DAPO vs. Base: Clean Colab Project

This English-only project compares Qwen3-1.7B-Base against DAPO-trained LoRA adapters on a fixed random 50-question MATH500 subset.

## Start here

1. Extract the new ZIP locally. Upload the **entire dapo-base-compare folder** to My Drive, replacing the previous project code. Keep the tests folder and setup_colab.py.
2. Open START_HERE.ipynb in Colab. Select a G4 or A100 GPU runtime.
3. Run cells in order. The first cell mounts Drive and copies code to /content/dapo-base-compare-clean.
4. The next cell installs a private environment and runs preflight checks. Do not install project requirements in the notebook kernel, import ML packages there, or mix cells from the old notebook.
5. Run the experiment, then the final cell to copy results to Drive.

The notebook ML scripts always run through /content/dapo-venv/bin/python. Colab's preinstalled torch, torchvision, torchaudio, gradio, diffusers, and gcsfs remain untouched. The isolated venv does not inherit system site-packages, PYTHONPATH, or user site-packages. No kernel restart is required for installation.

## Dependency checks

setup_colab.py creates an isolated venv using the Colab Python interpreter (supported versions: 3.10-3.13). It installs PyTorch 2.8.0 from the official CUDA 12.8 wheel index, followed by the pinned project requirements. It runs pip check and preflight.py before declaring the environment ready.

Preflight checks pinned package versions; Transformers and PEFT imports; absence of optional vision/audio packages; CUDA and BF16; a tiny randomly initialized Qwen3 LoRA forward/backward/generation on the GPU; and the complete unit-test suite. Any failed or skipped test stops setup. These checks do not download model weights.

This isolates the torchvision::nms conflict encountered with Colab's preinstalled packages. It does not guarantee compatibility with every future driver or runtime. Setup will show a clear failure before expensive training. Actual compatibility is verified on your selected GPU during preflight.

The venv is reused if Python and requirements match its saved fingerprint. To recover from an interrupted install, rerun setup. To use a fresh environment, pass --venv /content/dapo-venv-new and update PYTHON in the notebook. Do not reuse a manually modified environment with system packages enabled.

## Small experiment preset

The default notebook has no automatic wall-clock cutoff for setup, training, or evaluation. Training runs for the configured 20 rollout steps, then evaluates both models. Approximately two hours is a planning goal, not a deadline or a measured runtime guarantee. Live logs display elapsed time.

The preset uses 512 unique training questions, 20 rollout steps, four accepted groups per step, four sampled answers, and 2048-token completion limits (512-token overlong penalty buffer). Change `steps` and completion limits in config.json to adjust the workload. The last completed checkpoint is evaluated. No zero-update run is accepted as a DAPO result. Dynamic sampling can take longer when few groups contain mixed correct/incorrect answers; its candidate cap still reports failure if no valid group is found.

Optional CLI flags `--train-minutes` and `--budget-minutes` enable time limits only when explicitly supplied. The notebook supplies neither flag.

Only 50 evaluation questions are used; one question changes accuracy by two percentage points. A short training run can produce small, zero, or negative gains. This is a personal-project demonstration, not an exact reproduction of Ctrl-R Table 2 or strong evidence about reasoning capability.

## DAPO mechanisms

- Clip-Higher: asymmetric ratio clipping [1-0.20, 1+0.28].
- Dynamic Sampling: reject groups with all-correct or all-incorrect answers and resample until the effective batch is filled.
- Token-level policy loss: one common denominator over valid response tokens.
- Overlong handling: linear length penalty and loss masking for hard-truncated responses.

Rewards are +1/-1 for correctness plus length penalty. Advantages use group normalization. There is no KL penalty, entropy bonus, learned reward model, HMM, DFA, or Ctrl-R trajectory correction. Old-policy log probabilities remain fixed during two optimizer updates per rollout cycle.

Training generation uses temperature=1, top_p=1, top_k=0 to match the scored policy distribution. The four replies to each prompt are generated together. Different candidate questions and response backpropagation remain sequential. EOS padding is trimmed before scoring. This is a standalone BF16 LoRA implementation, not distributed verl/vLLM.

## Manual commands after setup

```bash
/content/dapo-venv/bin/python run_budget.py --eval-size 50
```

All commands must use the isolated interpreter, including preparation, training, evaluation, and tests. Do not use !python or sys.executable for project ML scripts: those refer to the notebook kernel. setup_colab.py is the only bootstrap script run with the kernel interpreter.

## Evaluation and saved outputs

Base and adapter evaluations share questions, model commit, prompt, grader, seed, generation limits, and package versions. compare.py refuses mismatched protocols or incomplete results.

The metric is answer accuracy, not classification precision. One sampled answer per question is used by default; multiple samples report average accuracy, not pass@k. Evaluation uses temperature=1, top_p=.95, top_k=20, and the same 2048-token budget for both models.

- runs/budget-*/budget_status.json: completion/failure/time-limit status and elapsed time.
- runs/budget-*/dapo/latest_checkpoint.json: last complete training checkpoint.
- runs/budget-*/dapo/train_log.jsonl: step timing, truncation, acceptance, and GPU memory.
- results/budget-*/base/predictions.jsonl and dapo/predictions.jsonl: answers and per-question correctness.
- results/budget-*/comparison/report.md: accuracy change in percentage points, paired bootstrap interval, and improved/regressed counts.

An interval crossing zero is inconclusive. It does not cover variability across independent training seeds. Hyperparameter selection needs a separate validation set; do not choose checkpoints using test scores. Checkpoints contain adapter, optimizer, and RNG states. Only load your own trainer.pt files.

## Data and limitations

Training source: EleutherAI/hendrycks_math, train split, Level 1-3. Evaluation source: HuggingFaceH4/MATH-500. Data preparation reads train-only Parquet shards and deduplicates training questions, freezes dataset revisions and JSONL hashes, removes whitespace-normalized exact matches against all 500 test questions, then freezes a seeded evaluation subset.

This does not detect paraphrases, integer-answer transformations, or pretraining contamination. Publication-quality claims need stronger decontamination and independent tests. The preset differs from the paper in LoRA updates, batch/data size, training steps, completion lengths, and checkpoint selection. config_full.json retains the previous larger settings and is intended for a larger experiment.

If dynamic sampling cannot fill a batch, inspect outputs and grading before spending more GPU time. A weak base model with short responses may produce mostly incorrect groups. Removing filtering changes DAPO. Adding SFT warmup requires a base/SFT-only/SFT+DAPO comparison to isolate RL gains.

## Validation status

Local Python and notebook syntax, isolated-environment setup logic, budget-stop behavior, archive structure, and CPU tests were checked. The delivery host has no CUDA/PyTorch/math-verify; actual wheel installation and GPU training were not performed there. Colab preflight must pass all dependency, GPU, and unit checks before training. No measured model accuracy is included and no gain is guaranteed.

## References

- [DAPO paper](https://arxiv.org/abs/2503.14476)
- [Ctrl-R paper](https://arxiv.org/abs/2603.01641)
- [Official CUDA wheels](https://download.pytorch.org/whl/cu128)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)

Follow upstream dependency, dataset, and model licenses.

## Live progress

The experiment cell now forwards merged stdout/stderr from all child scripts, saves console.log inside the run folder and notebook_console.log at the project root, and emits a heartbeat every 30 seconds. Stage banners identify data preparation, training, base evaluation, adapter evaluation, and reporting. Training prints candidate groups, individual answer lengths/correctness, and update epochs before the per-step summary. A heartbeat confirms that the process is running; it does not prove useful training progress. Interrupting the notebook runner terminates its process group. Live forwarding cannot be retroactively attached to a previously launched run.

## G4 batched preset

Version: MATH-L123-G4-2048-v3. Training uses 20 steps, four accepted question groups per step, and four replies generated together per candidate question. Both evaluation response limits remain 2048 tokens. There is no default setup, training, or experiment time cutoff. Evaluation remains the same 50-question subset for comparability.

The candidate cap is 128 groups per step (a sampling failure guard, not a time limit). This is more training work than the previous one-group preset; batching does not guarantee a shorter complete run or higher accuracy. Gradient accumulation keeps one optimizer update per epoch, with one token denominator across all four accepted groups.

rollout_audit.jsonl records questions, gold answers, complete responses, extracted answers, and grading results for every sampled reply. train_log.jsonl adds acceptance rate, mean reward, missing-answer rate, sampling throughput, pre-clipping gradient norms, and token-weighted mean absolute policy-ratio deviation before each update. These help distinguish slow generation, difficult questions, formatting failures, and policy movement. No automatic difficulty filtering is applied; inspect the audit before changing training data. Do not select training data using evaluation questions.

Use a new run for this preset; old one-group checkpoints have a different training configuration. Lower num_generations if a batched rollout runs out of memory (keep at least two). This preset still requires BF16 and is not configured for T4.

Training now prints each question, dataset gold answer, four complete responses, extracted final answers, correctness, truncation, and group acceptance/rejection. Batched responses appear after the generation call completes. Gold answers are displayed for inspection and grading only; they are not added to model prompts.

## MATH Level 1-3 training preset

This version replaces DAPO-Math-17K with EleutherAI/hendrycks_math train-only shards across all seven subjects. It filters to Level 1-3, extracts the last balanced boxed gold answer from the solution, removes exact matches to all 500 MATH500 questions, and randomly selects 512 eligible unique training rows using the configured seed. Dataset commits, selection metadata, and file hashes are saved. Solutions are not included in policy prompts.

After setup, the notebook has one code cell evaluating the original base on 10 fixed questions from the seeded 50-question MATH500 subset. It prints accuracy and truncation rate and saves full predictions to Drive. This diagnostic is distinct from training-data success rate and does not select training data. The final comparison still evaluates both models on all 50 selected questions. Ten questions are too few to establish a reliable improvement.

Keep 2048-token limits, four batched replies/question, four accepted groups/step, 20 steps, verbose rollout logs, and no default time cutoff. Start a fresh run when changing training datasets.
