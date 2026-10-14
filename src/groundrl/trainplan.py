"""Training data records and the dry-run plan. Needs no torch, no tokenizer download.

``build_plan`` loads the real dataset, renders every prompt, counts tokens with a tokenizer
(the dependency-free :class:`StubTokenizer` by default, a Hugging Face tokenizer when one is
passed in), and reports what a run would do: how many examples survive the prompt budget, how
many optimizer steps there are, and a rough memory estimate. The estimate is a back-of-envelope
formula, not a measurement, and the plan says so.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .dataset import load_dataset
from .example import Example, make_examples
from .prompts import PROMPT_VERSION, build_messages, render_chat_stub

_PIECE = re.compile(r"0[xX][0-9a-fA-F]+|\d+|[A-Za-z_]+|[^\w\s]")


class StubTokenizer:
    """A crude BPE-like token counter: words ~5 chars/token, digits ~3, hex ~2, symbols 1.

    Real tokenizers differ (typically by 10-30% on this kind of text). It exists so that dry runs
    and tests need no download; training always uses the model's own tokenizer.
    """

    def count(self, text: str) -> int:
        total = 0
        for piece in _PIECE.findall(text):
            if piece[:2].lower() == "0x":
                total += max(1, math.ceil((len(piece) - 2) / 2)) + 1
            elif piece.isdigit():
                total += math.ceil(len(piece) / 3)
            elif piece[0].isalpha() or piece[0] == "_":
                total += max(1, math.ceil(len(piece) / 5))
            else:
                total += 1
        return total

    def render(self, messages: Sequence[dict[str, str]], enable_thinking: bool = False) -> str:
        return render_chat_stub(messages, add_generation_prompt=True,
                                enable_thinking=enable_thinking)


def prompt_text(ex: Example, render: Callable[[Sequence[dict[str, str]]], str]) -> str:
    return render(build_messages(ex))


def sft_record(ex: Example, prompt: str) -> dict[str, str]:
    """Prompt/completion pair for SFTTrainer. TRL appends the end-of-sequence token itself."""
    return {"prompt": prompt, "completion": ex.task.gold_answer}


def grpo_record(ex: Example, prompt: str) -> dict[str, str]:
    """GRPO row: the rendered prompt plus the example JSON the reward function reads."""
    return {"prompt": prompt, "example_json": ex.to_json()}


# (layers, hidden size, KV width = kv heads x head dim), vocabulary size. Used by the estimate only.
_ARCH = {0.6: (28, 1024, 1024), 1.7: (28, 2048, 1024), 4.0: (36, 2560, 1024), 8.0: (36, 4096, 1024)}
_VOCAB = 151_936


def estimate_memory_gb(
    params_b: float, load_in_4bit: bool, lora_r: int, prompt_tokens: int, completion_tokens: int,
    micro_batch: int, stage: str, gen_batch: int = 0,
) -> float:
    """A rough peak-GPU-memory estimate in GB. A formula, NOT a measurement.

    Terms: weights (2 bytes/param, 0.55 in 4-bit), LoRA parameters + gradients + Adam states
    (16 bytes each), checkpointed activations (layer boundaries plus one layer of workspace),
    the logits tensor (about 10 bytes per position x vocabulary: bf16 logits, fp32 upcast and its
    gradient), the KV cache of Hugging Face generation during GRPO, and 1.5 GB of CUDA context.
    Training and generation phases are sequential, so the larger of the two counts. SFT computes
    logits for every position, GRPO only for completion tokens. The Qwen3 vocabulary (151,936) makes
    the logits term the one that most often decides whether a run fits.
    """
    nearest = min(_ARCH, key=lambda k: abs(k - params_b))
    layers, hidden, kv_dim = _ARCH[nearest]
    weights = params_b * 1e9 * (0.55 if load_in_4bit else 2.0)
    lora = lora_r * 0.012 * params_b * 1e9 / 16 * 16
    seq = prompt_tokens + completion_tokens
    tokens = micro_batch * seq
    acts = tokens * layers * hidden * 2 + tokens * hidden * 34 * 2
    logit_positions = tokens if stage == "sft" else micro_batch * completion_tokens
    logits = logit_positions * _VOCAB * 10
    train_phase = acts + logits
    gen_phase = 0.0
    if stage == "grpo" and gen_batch:
        gen_phase = gen_batch * seq * layers * 2 * kv_dim * 2 + gen_batch * _VOCAB * 4
    if stage == "eval":
        train_phase = 0.0
        gen_phase = micro_batch * seq * layers * 2 * kv_dim * 2 + micro_batch * _VOCAB * 4
    return (weights + lora + max(train_phase, gen_phase)) / 1e9 * 1.1 + 1.5


_PARAMS_B = (("0.6b", 0.6), ("1.7b", 1.7), ("4b", 4.0), ("8b", 8.0))


def model_params_b(name: str) -> float:
    low = name.lower()
    for tag, size in _PARAMS_B:
        if tag in low:
            return size
    return 1.0


def _stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0}
    return {"mean": round(mean(values), 1), "p95": float(np.percentile(values, 95)),
            "max": int(max(values))}


def build_plan(
    cfg: dict[str, Any], stage: str, tokenizer: StubTokenizer | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The plan for ``stage`` in {"sft", "grpo", "eval"}; raises if the data cannot be loaded."""
    if stage not in ("sft", "grpo", "eval"):
        raise ValueError(f"unknown stage {stage!r}")
    tok = tokenizer or StubTokenizer()
    data = cfg["data"]
    root = Path(repo_root) if repo_root else Path.cwd()
    data_dir = Path(data["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else root / data_dir
    ds = load_dataset(data_dir)
    split = cfg["eval"]["split"] if stage == "eval" else data["train_split"]
    tasks = ds.tasks[split][: data["limit_train"] if stage != "eval" else None]
    examples = make_examples(tasks, ds.index)
    render = lambda msgs: tok.render(msgs, data["enable_thinking"])  # noqa: E731
    prompt_tokens = [tok.count(prompt_text(e, render)) for e in examples]
    completion_tokens = [tok.count(e.task.gold_answer) + 1 for e in examples]
    keep = [i for i, n in enumerate(prompt_tokens) if n <= data["max_prompt_tokens"]]
    dropped = len(examples) - len(keep)
    plan: dict[str, Any] = {
        "stage": stage, "config": cfg["name"], "split": split, "prompt_version": PROMPT_VERSION,
        "tokenizer": type(tok).__name__ + (" (approximate counts)" if isinstance(
            tok, StubTokenizer) else ""),
        "examples": len(examples), "kept_after_prompt_budget": len(keep), "dropped": dropped,
        "prompt_tokens": _stats(prompt_tokens),
        "oracle_completion_tokens": _stats(completion_tokens),
        "max_prompt_tokens": data["max_prompt_tokens"],
        "max_completion_tokens": data["max_completion_tokens"],
        "unanswerable_share": round(sum(not e.task.answerable for e in examples)
                                    / max(1, len(examples)), 3),
        "warnings": [],
        "notes": [],
    }
    warn = plan["warnings"]
    if examples and dropped / len(examples) > 0.10:
        warn.append(f"{dropped} of {len(examples)} prompts exceed max_prompt_tokens "
                    f"({data['max_prompt_tokens']}); raise it")
    elif dropped and stage != "eval":
        plan["notes"].append(f"{dropped} of {len(examples)} prompts exceed max_prompt_tokens and "
                             "are dropped from training (evaluation never drops any)")
    if plan["oracle_completion_tokens"]["max"] * 1.2 > data["max_completion_tokens"]:
        warn.append("the longest gold answer (stub estimate x1.2) does not fit "
                    "max_completion_tokens; real completions may be truncated")
    kept = max(1, len(keep))
    params = model_params_b(cfg["model"]["name_or_path"])
    gen_batch = 0
    if stage == "sft":
        a = cfg["sft"]["args"]
        per = a.get("per_device_train_batch_size", 8) * a.get("gradient_accumulation_steps", 1)
        steps_epoch = math.ceil(kept / per)
        epochs = a.get("num_train_epochs", 1)
        plan.update({"effective_batch": per, "steps_per_epoch": steps_epoch,
                     "planned_steps": a.get("max_steps", -1) if a.get("max_steps", -1) > 0
                     else math.ceil(steps_epoch * epochs), "epochs": epochs})
        micro = a.get("per_device_train_batch_size", 8)
    elif stage == "grpo":
        a = cfg["grpo"]["args"]
        n = a["num_generations"]
        micro = a.get("per_device_train_batch_size", 8)
        accum = a.get("gradient_accumulation_steps", 1)
        completions_per_step = micro * accum
        prompts_per_step = completions_per_step // n
        steps_per_gen = a.get("steps_per_generation") or accum
        gen_batch = micro * steps_per_gen
        steps = a.get("max_steps", -1)
        plan.update({
            "group_size": n, "completions_per_optimizer_step": completions_per_step,
            "prompts_per_optimizer_step": prompts_per_step, "planned_steps": steps,
            "epochs_covered": round(steps * prompts_per_step / kept, 2) if steps > 0 else None,
            "kl_beta": a.get("beta", 0.0), "learning_rate": a.get("learning_rate"),
            "reward": cfg["grpo"]["reward"], "use_vllm": bool(a.get("use_vllm", False)),
            "generated_tokens_per_step": completions_per_step * data["max_completion_tokens"],
            "generation_batch": gen_batch, "steps_per_generation": steps_per_gen,
        })
        if steps_per_gen > accum:
            warn.append("steps_per_generation exceeds gradient_accumulation_steps: updates reuse "
                        "older generations (slightly off-policy)")
        if prompts_per_step < 1:
            warn.append("fewer than one prompt per optimizer step; raise gradient accumulation")
    else:
        micro = cfg["eval"]["batch_size"]
        plan.update({"batch_size": micro, "max_new_tokens": cfg["eval"]["max_new_tokens"],
                     "temperature": cfg["eval"]["temperature"]})
    est = estimate_memory_gb(params, cfg["model"]["load_in_4bit"], cfg["lora"]["r"],
                             data["max_prompt_tokens"], data["max_completion_tokens"], micro,
                             stage, gen_batch)
    plan["memory_estimate_gb"] = round(est, 1)
    plan["memory_estimate_note"] = ("back-of-envelope formula, NOT measured; real usage depends "
                                    "on the TRL/transformers version and the GPU")
    vram = cfg["profile"]["vram_gb"]
    plan["profile_vram_gb"] = vram
    if est > 0.85 * vram:
        warn.append(f"estimated {est:.1f} GB is above 85% of the {vram} GB profile; reduce "
                    "per_device_train_batch_size, steps_per_generation or num_generations")
    return plan


def format_plan(plan: dict[str, Any]) -> str:
    lines = [f"DRY RUN: {plan['stage']} / {plan['config']} (no model loaded, no GPU, no torch)"]
    skip = {"warnings", "notes", "stage", "config", "memory_estimate_note"}
    for key, value in plan.items():
        if key not in skip:
            lines.append(f"  {key}: {value}")
    lines.append(f"  note: {plan['memory_estimate_note']}")
    lines.extend(f"  note: {n}" for n in plan["notes"])
    if plan["warnings"]:
        lines.extend(f"  WARNING: {w}" for w in plan["warnings"])
    else:
        lines.append("  no warnings")
    return "\n".join(lines)
