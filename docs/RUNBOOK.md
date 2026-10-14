# Runbook: training and evaluating a groundrl model

**Status: SFT has been run on gfx906; GRPO has not.** The first real SFT run on Qwen3-0.6B is
committed under `results/groundrl-sft/` (`eval_baseline.json`, `eval_sft.json`, `summary_sft.json`,
`LEAKAGE_CHECK.md`). The GRPO row in the README still reads NOT YET RUN. Hyperparameters in
`train/configs/*.json` for the configs that have not been exercised are still untuned defaults,
their memory figures are formulas rather than measurements, and no timing or cost estimate is
given for them. What is verified without a GPU: the configs validate, the keyword arguments they
pass exist in the pinned TRL / transformers / PEFT sources (checked by `tests/test_trainconf.py`
against `tests/api_fields.json`), the dry run works without torch, and the whole control flow
runs against a fake stack (`--mock`, labelled MOCK).

Training results are not claimed in the README or REPORT until `results/**/eval_*.json` files
from real runs exist; `scripts/make_report.py` renders only from those files.

## 1. Pick a config

| config | model | VRAM profile | notes |
|---|---|---|---|
| `qwen3-0.6b-24gb` | Qwen3-0.6B, bf16 + LoRA | 24 GB | group size 8, prompt budget 2048 |
| `qwen3-0.6b-12gb` | Qwen3-0.6B, bf16 + LoRA | 12 GB | group size 4, prompt budget 1536 |
| `qwen3-1.7b-24gb` | Qwen3-1.7B, bf16 + LoRA | 24 GB | group size 8, micro-batches of 2 |
| `qwen3-1.7b-12gb` | Qwen3-1.7B, 4-bit NF4 + LoRA (QLoRA) | 12 GB | needs `bitsandbytes` |

Run `python train/grpo.py --config train/configs/<name>.json --dry-run` first. It prints the
estimated memory for the stage; if it is above 85% of the profile it prints a warning. The Qwen3
vocabulary is large (151,936), and the logits tensor is usually what decides whether a run fits:
lower `per_device_train_batch_size` before anything else.

## 2. Install

### Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# a CUDA build of PyTorch that matches your driver (see pytorch.org for the exact command)
pip install torch --index-url https://download.pytorch.org/whl/<cuda-tag>   # tag from pytorch.org, e.g. cu124
# this package with the training extra
pip install -e ".[train,dev]"
# 4-bit profile only
pip install bitsandbytes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # use the tag pytorch.org shows for your driver
pip install -e ".[train,dev]"
pip install bitsandbytes               # 4-bit profile only
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$env:PYTHONUTF8 = "1"                  # avoid cp1252 errors when reading the corpus
$env:HF_HOME = "D:\hf-cache"           # keep model downloads off a small system drive
```

Windows notes:

* vLLM does not run natively on Windows. Leave `use_vllm` false (the shipped default) or use WSL2.
* Every script uses `dataset_num_proc: 1`, which avoids the slow multiprocessing spawn on Windows.
* Long paths: enable them (`git config --system core.longpaths true`) if you clone into a deep directory.
* Line continuation in PowerShell is the backtick; the commands below are written one per line.

## 3. Smoke tests that need no GPU

```bash
pytest -q                                                  # whole suite, no torch needed
python train/sft.py  --config train/configs/qwen3-0.6b-24gb.json --dry-run
python train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --dry-run
python train/eval.py --config train/configs/qwen3-0.6b-24gb.json --dry-run
python train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --mock --limit 16
```

`--mock` runs the real control flow against a fake model and fake trainers. It prints a MOCK
banner, writes under `outputs/mock/` with `"mock": true`, and `scripts/make_report.py` never
counts such files.

## 4. The pipeline

```bash
# data (already committed; these reproduce it)
python scripts/fetch_corpus.py                  # optional re-download, writes data/MANIFEST.json
groundrl build-tasks                            # deterministic, seed 20260
groundrl check-data

# reference points that need no model
python scripts/reward_audit.py --split test     # reward-hacking audit
groundrl baseline-eval --split test             # oracle / extractive / shortcut / refuse-all

# stage 1: SFT with LoRA, then merge the adapter into the base weights
python train/sft.py --config train/configs/qwen3-0.6b-24gb.json --out outputs/sft

# stage 2: GRPO with reward v2 on a fresh LoRA over the merged SFT model
python train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --init-from outputs/sft/merged --out outputs/grpo

# evaluation on TEST (greedy decoding), one result file per run
python train/eval.py --config train/configs/qwen3-0.6b-24gb.json --label base
python train/eval.py --config train/configs/qwen3-0.6b-24gb.json --model outputs/sft/merged --label sft
python train/eval.py --config train/configs/qwen3-0.6b-24gb.json --model outputs/sft/merged --adapter outputs/grpo/adapter --label grpo-v2

# report from results/*.json only
python scripts/make_report.py
```

On Windows PowerShell the commands are the same: forward-slash paths work there too, and each
command goes on one line. `--init-from` points GRPO at the merged SFT model, so the KL reference (the model with
the adapter disabled) is the SFT policy, which is what TRL's GRPOTrainer expects when you start
from a trained adapter.

Useful flags on every stage: `--seed`, `--lr`, `--max-len` (total prompt + answer tokens),
`--limit N` (first N examples), `--set dotted.path=json` for anything else. GRPO also takes
`--group-size` and `--kl-beta`. Example: `--group-size 4 --kl-beta 0.05 --seed 1`.

## 5. What to watch while GRPO runs

TRL logs these every step (`logging_steps: 1`); the reward wrapper adds the `groundrl/*` ones.

| metric | healthy | trouble |
|---|---|---|
| `reward`, `rewards/groundrl_reward_v2/mean` | rises from a low start | flat at 0: see troubleshooting |
| `reward_std`, `frac_reward_zero_std` | std > 0, few zero-std groups | many zero-std groups mean no learning signal |
| `groundrl/verified_rate` | rises | rises while `groundrl/correctness` falls: hacking |
| `groundrl/correctness` | rises | stalls near 0 |
| `groundrl/abstain_rate` | near the unanswerable share (about 11%) | climbs towards 1: collapse to refusal |
| `groundrl/hallucinated_citation_rate` | falls towards 0 | stays high: citation format not learned |
| `groundrl/mean_words`, `completions/mean_length` | stable | grows without bound |
| `completions/clipped_ratio` | near 0 | high: raise `max_completion_length` |
| `kl` | small and slowly growing | jumps by orders of magnitude: lower the learning rate |

The decision tree for these symptoms is in `docs/FLOWCHARTS.md`.

## 6. Renting a GPU

* Pick the 24 GB profile and a single consumer or workstation card (RTX 3090/4090, A5000, L4).
  The 12 GB profiles exist for smaller cards you may already own.
* Any provider with a PyTorch/CUDA image works (Vast.ai, RunPod, Lambda, and similar). Use a
  persistent volume for `HF_HOME` and `outputs/`, run inside `tmux` or `screen`, and shut the
  instance down when the run finishes. Prices and run times were not measured here; the dry run
  tells you the number of optimizer steps, and a 5-step trial run (`--set grpo.args.max_steps=5`)
  tells you the seconds per step on the card you rented, which is the only number worth trusting.
* Commit only `results/*.json` (and the markdown report). Never commit `outputs/`: adapters and
  merged weights are large and are in `.gitignore`.
* Resume an interrupted run with `--set grpo.args.resume_from_checkpoint="outputs/grpo/trainer/checkpoint-100"`
  (`resume_from_checkpoint` is a `TrainingArguments` field).
* The Qwen3 models are public; no Hugging Face token is needed to download them.

## 7. Reading the results honestly

* Compare against the scripted baselines first. `extractive` (locate and copy the best line)
  scores 0.613 reward v2 and 57.4% grounded-correct on TEST; a trained model that does not beat it
  has not learned more than lexical matching.
* Use `eval_*.json`, not the training reward, to judge a run. Rewards of different versions are
  not comparable with each other.
* One seed is an anecdote. Run at least three seeds before claiming a difference, and report the
  bootstrap intervals the evaluation already computes.
* Do not tune on TEST. Use VAL (`--split val`) while iterating.
* Rerun `python scripts/reward_audit.py` if you change the reward, and read the exploitable list
  before you read the learning curve.
