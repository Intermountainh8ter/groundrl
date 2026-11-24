# GRPO run — degenerate (null result)

**Date:** 2026-10-14 · **Config:** `train/configs/qwen3-0.6b-gfx906.json` ·
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
GRPO config files (`train/configs/qwen3-0.6b-gfx906.json`, which ran, and
`train/configs/qwen3-0.6b-24gb.json`, which the pre-registration names) are unchanged.

**Deviation from the pre-registration.** The run used `train/configs/qwen3-0.6b-gfx906.json` (the config recorded in `summary_grpo.json`), not the `qwen3-0.6b-24gb.json` file the pre-registration names. The two files differ only in dtype: the gfx906 one trains in float16 (`fp16: true`, `bf16: false`, `dtype: float16`) because the card has no bfloat16 hardware (`docs/ROCM_GFX906.md`). That is a deviation from the pre-registered config, and it was not itself pre-registered. Both files are unchanged since the commit that added the gfx906 one (`f411923`), which precedes the run.

Fixing the degeneracy is a hypothesis-driven *new* run with its own pre-registration:
see `docs/METHODOLOGY.md` §6a for the four hypotheses (prompt-template mismatch,
max-new-tokens cap, EOS-token drift, float16 instead of the pre-registered bfloat16).

## Why the adapter is not committed

The adapter weights at `/home/ammo/bench/results/groundrl-grpo/adapter/` were
written by the trainer but received zero gradient updates. They are byte-near
to the init distribution and have no training signal to preserve. The LoRA
`.safetensors` is gitignored under the standard training-outputs rule.
