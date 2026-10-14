# GRPO run — degenerate (null result)

**Date:** 2026-10-14 · **Config:** `train/configs/qwen3-0.6b-24gb.json` ·
**Base:** `results/groundrl-sft/merged` · **Steps:** 300 ·
**Wall-clock:** 21,350 s (5h 55m) · **Git:** `bcf650b`

## What was observed

Every log entry across 300 optimizer steps reported:

| metric | value |
|---|---|
| `rewards/groundrl_reward_v2/mean` | 0.0 |
| `rewards/groundrl_reward_v2/std` | 0.0 |
| `completions/mean_length` | 0.0 |
| `completions/max_length` | 0.0 |
| `completions/clipped_ratio` | 1.0 |
| `grad_norm` | 0.0 |
| `kl` | NaN |
| `loss` | 0.0 |
| `train_loss` | 0.0 |

Full log history is in `summary_grpo.json`.

## Cause (from stdout trace)

The merged SFT policy, when given the GRPO prompt rendering, emits only the Qwen3
chat-template no-think shortcut and then stops:

```
<|im_start|>assistant
<think>

</think>


```

No new tokens → no reward signal → no gradient → no update, run after run.

## Why we are not re-tuning in place

The pre-registration (`docs/METHODOLOGY.md` §6) specifies these settings. Changing
them after seeing results is exactly the thing the pre-registration forbids. The
GRPO config file (`train/configs/qwen3-0.6b-24gb.json`) is unchanged.

Fixing the degeneracy is a hypothesis-driven *new* run with its own pre-registration:
see `docs/METHODOLOGY.md` §6a for the three leading hypotheses (prompt-template
mismatch, max-new-tokens cap, EOS-token drift).

## Why the adapter is not committed

The adapter weights at `/home/ammo/bench/results/groundrl-grpo/adapter/` were
written by the trainer but received zero gradient updates. They are byte-near
to the init distribution and have no training signal to preserve. The LoRA
`.safetensors` is gitignored under the standard training-outputs rule.
