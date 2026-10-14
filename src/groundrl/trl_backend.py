"""Training and evaluation runners built on TRL, PEFT and transformers.

Everything heavy goes through a ``stack`` namespace returned by :func:`load_stack`, which imports
torch/transformers/trl/peft/datasets lazily. Nothing in this module imports them at import time,
and the test suite exercises the whole control flow with the fake stack from
:mod:`groundrl.mockstack` (results of such runs are labelled MOCK and carry ``"mock": true``).

STATUS: the code paths below have not been executed against real torch/TRL in this repository's
development environment (it has no GPU). The keyword arguments passed to ``SFTConfig``,
``GRPOConfig`` and ``LoraConfig`` are checked against the field names of the pinned library
sources by the test suite; behaviour on real hardware is unverified.

Flow: SFT (LoRA) -> merge the adapter into the base weights -> GRPO (fresh LoRA on the merged
model, so the KL reference is the SFT policy) -> evaluation. This follows TRL's own guidance for
GRPO on top of a trained adapter.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .dataset import load_dataset
from .evaluate import evaluate_completions
from .example import Example, make_examples
from .prompts import PROMPT_VERSION, build_messages
from .rewards import make_trl_reward
from .trainplan import grpo_record, sft_record


def load_stack() -> SimpleNamespace:  # pragma: no cover - needs the optional training extra
    """Import the training stack lazily; raise a helpful error if the extra is missing."""
    try:
        import datasets
        import peft
        import torch
        import transformers
        import trl
    except ImportError as exc:
        raise ImportError(
            "training needs the optional extra: pip install -e '.[train]' "
            f"(missing: {exc.name}). Dry runs and tests do not need it."
        ) from exc
    return SimpleNamespace(
        torch=torch, transformers=transformers, trl=trl, peft=peft, datasets=datasets,
        AutoModelForCausalLM=transformers.AutoModelForCausalLM,
        AutoTokenizer=transformers.AutoTokenizer,
        BitsAndBytesConfig=transformers.BitsAndBytesConfig,
        set_seed=transformers.set_seed,
        LoraConfig=peft.LoraConfig, PeftModel=peft.PeftModel,
        SFTConfig=trl.SFTConfig, SFTTrainer=trl.SFTTrainer,
        GRPOConfig=trl.GRPOConfig, GRPOTrainer=trl.GRPOTrainer,
        Dataset=datasets.Dataset,
    )


# ---------------------------------------------------------------------------------------------
# helpers


def _dtype(stack: SimpleNamespace, name: str) -> Any:
    return getattr(stack.torch, name)


def load_tokenizer(stack: SimpleNamespace, cfg: dict[str, Any], path: str | None = None) -> Any:
    tok = stack.AutoTokenizer.from_pretrained(
        path or cfg["model"]["init_from"] or cfg["model"]["name_or_path"],
        trust_remote_code=cfg["model"]["trust_remote_code"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(stack: SimpleNamespace, cfg: dict[str, Any], path: str | None = None) -> Any:
    m = cfg["model"]
    dtype = _dtype(stack, m["dtype"])
    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": m["attn_implementation"],
                              "trust_remote_code": m["trust_remote_code"]}
    if m["load_in_4bit"]:
        kwargs["quantization_config"] = stack.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    return stack.AutoModelForCausalLM.from_pretrained(
        path or m["init_from"] or m["name_or_path"], **kwargs)


def lora_config(stack: SimpleNamespace, cfg: dict[str, Any]) -> Any:
    lo = cfg["lora"]
    return stack.LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                            target_modules=lo["target_modules"], bias=lo["bias"],
                            task_type="CAUSAL_LM")


def render_prompt(tok: Any, ex: Example, enable_thinking: bool) -> str:
    return tok.apply_chat_template(build_messages(ex), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)


def build_examples(cfg: dict[str, Any], split: str, limit: int | None = None,
                   repo_root: Path | None = None) -> list[Example]:
    root = Path(repo_root) if repo_root else Path.cwd()
    data_dir = Path(cfg["data"]["data_dir"])
    ds = load_dataset(data_dir if data_dir.is_absolute() else root / data_dir)
    return make_examples(ds.tasks[split][:limit], ds.index)


def prompt_budget_filter(tok: Any, prompts: list[str], max_tokens: int) -> list[int]:
    """Indices of prompts that fit the budget (the tokenizer's own count)."""
    return [i for i, p in enumerate(prompts)
            if len(tok(p, add_special_tokens=False)["input_ids"]) <= max_tokens]


def _require_rows(rows: list[Any], budget: int) -> None:
    if not rows:
        raise ValueError(f"no training prompt fits max_prompt_tokens={budget}; raise the budget")


def environment_info(stack: SimpleNamespace) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets"):
        info[name] = getattr(getattr(stack, name, None), "__version__", "unknown")
    try:
        if stack.torch.cuda.is_available():
            props = stack.torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        else:
            info["gpu"] = None
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        info["gpu"] = "unknown"
    return info


def git_commit(root: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _numeric_log(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [{k: v for k, v in row.items() if isinstance(v, int | float)} for row in history]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_adapter(stack: SimpleNamespace, cfg: dict[str, Any], adapter_dir: Path, out_dir: Path,
                  tok: Any) -> Path:
    """Merge a LoRA adapter into a full-precision copy of the base model and save it.

    The base is reloaded unquantised (on CPU) even for 4-bit runs, because merging into a
    quantised model is lossy.
    """
    base = stack.AutoModelForCausalLM.from_pretrained(
        cfg["model"]["init_from"] or cfg["model"]["name_or_path"],
        dtype=_dtype(stack, cfg["model"]["dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"])
    merged = stack.PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


# ---------------------------------------------------------------------------------------------
# SFT


def run_sft(
    cfg: dict[str, Any], out_dir: Path, stack: SimpleNamespace | None = None, mock: bool = False,
    repo_root: Path | None = None, merge: bool = True,
) -> dict[str, Any]:
    """Supervised fine-tuning on gold answers (LoRA). Writes the adapter and a summary JSON."""
    stack = stack or load_stack()
    out_dir = Path(out_dir)
    sft = cfg["sft"]["args"]
    seed = sft.get("seed", 0)
    stack.set_seed(seed)
    tok = load_tokenizer(stack, cfg)
    train = build_examples(cfg, cfg["data"]["train_split"], cfg["data"]["limit_train"], repo_root)
    evals = build_examples(cfg, cfg["data"]["eval_split"], cfg["data"]["limit_eval"], repo_root)
    thinking = cfg["data"]["enable_thinking"]
    budget = cfg["data"]["max_prompt_tokens"]

    def records(examples: list[Example]) -> list[dict[str, str]]:
        prompts = [render_prompt(tok, e, thinking) for e in examples]
        keep = prompt_budget_filter(tok, prompts, budget)
        return [sft_record(examples[i], prompts[i]) for i in keep]

    train_rows, eval_rows = records(train), records(evals)
    _require_rows(train_rows, budget)
    args = stack.SFTConfig(output_dir=str(out_dir / "trainer"), **sft)
    trainer = stack.SFTTrainer(
        model=load_model(stack, cfg), args=args, processing_class=tok,
        train_dataset=stack.Dataset.from_list(train_rows),
        eval_dataset=stack.Dataset.from_list(eval_rows), peft_config=lora_config(stack, cfg))
    started = time.time()
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    merged_dir = merge_adapter(stack, cfg, adapter_dir, out_dir / "merged", tok) if merge else None
    summary = {
        "kind": "train", "stage": "sft", "mock": mock, "config": cfg["name"], "seed": seed,
        "prompt_version": PROMPT_VERSION, "train_examples": len(train_rows),
        "dropped_by_prompt_budget": len(train) - len(train_rows),
        "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir) if merged_dir else None,
        "seconds": round(time.time() - started, 1), "log_history": _numeric_log(
            trainer.state.log_history),
        "environment": environment_info(stack), "git_commit": git_commit(repo_root),
    }
    _write_json(out_dir / "summary_sft.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------
# GRPO


def run_grpo(cfg: dict[str, Any], out_dir: Path, stack: SimpleNamespace | None = None,
             mock: bool = False, repo_root: Path | None = None) -> dict[str, Any]:
    """GRPO with the groundrl reward on a fresh LoRA. Writes a summary JSON with the full log."""
    stack = stack or load_stack()
    out_dir = Path(out_dir)
    g = cfg["grpo"]["args"]
    seed = g.get("seed", 0)
    stack.set_seed(seed)
    tok = load_tokenizer(stack, cfg)
    train = build_examples(cfg, cfg["data"]["train_split"], cfg["data"]["limit_train"], repo_root)
    thinking = cfg["data"]["enable_thinking"]
    prompts = [render_prompt(tok, e, thinking) for e in train]
    keep = prompt_budget_filter(tok, prompts, cfg["data"]["max_prompt_tokens"])
    rows = [grpo_record(train[i], prompts[i]) for i in keep]
    _require_rows(rows, cfg["data"]["max_prompt_tokens"])
    args = stack.GRPOConfig(output_dir=str(out_dir / "trainer"),
                            **{"remove_unused_columns": False, **g})
    trainer = stack.GRPOTrainer(
        model=load_model(stack, cfg), reward_funcs=make_trl_reward(cfg["grpo"]["reward"]),
        args=args, train_dataset=stack.Dataset.from_list(rows), processing_class=tok,
        peft_config=lora_config(stack, cfg))
    started = time.time()
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    summary = {
        "kind": "train", "stage": "grpo", "mock": mock, "config": cfg["name"], "seed": seed,
        "reward": cfg["grpo"]["reward"], "prompt_version": PROMPT_VERSION,
        "group_size": g["num_generations"], "kl_beta": g.get("beta", 0.0),
        "learning_rate": g.get("learning_rate"), "train_examples": len(rows),
        "dropped_by_prompt_budget": len(train) - len(rows), "adapter_dir": str(adapter_dir),
        "seconds": round(time.time() - started, 1),
        "log_history": _numeric_log(trainer.state.log_history),
        "environment": environment_info(stack), "git_commit": git_commit(repo_root),
    }
    _write_json(out_dir / "summary_grpo.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------
# evaluation


def generate_completions(stack: SimpleNamespace, model: Any, tok: Any, prompts: list[str],
                         batch_size: int, max_new_tokens: int, temperature: float) -> list[str]:
    """Batched generation with left padding; returns only the newly generated text."""
    tok.padding_side = "left"
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens,
                                      "pad_token_id": tok.pad_token_id}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with stack.torch.no_grad():
            ids = model.generate(**enc, **gen_kwargs)
        new = ids[:, enc["input_ids"].shape[1]:]
        outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outputs


def run_eval(cfg: dict[str, Any], model_path: str | None, adapter_dir: str | None,
             out_path: Path, split: str | None = None, stack: SimpleNamespace | None = None,
             mock: bool = False, limit: int | None = None, repo_root: Path | None = None,
             label: str | None = None) -> dict[str, Any]:
    """Generate answers for a split with a trained model and score them. Writes the JSON."""
    stack = stack or load_stack()
    ev = cfg["eval"]
    split = split or ev["split"]
    path = model_path or cfg["model"]["init_from"] or cfg["model"]["name_or_path"]
    tok = load_tokenizer(stack, cfg, path)
    model = load_model(stack, cfg, path)
    if adapter_dir:
        model = stack.PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    examples = build_examples(cfg, split, limit, repo_root)
    prompts = [render_prompt(tok, e, cfg["data"]["enable_thinking"]) for e in examples]
    started = time.time()
    completions = generate_completions(stack, model, tok, prompts, ev["batch_size"],
                                       ev["max_new_tokens"], ev["temperature"])
    token_counts = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in completions]
    result = evaluate_completions(examples, completions, ev["bootstrap"],
                                  cfg["grpo"]["args"].get("seed", 0), token_counts)
    result.update({
        "kind": "eval", "mock": mock, "config": cfg["name"], "label": label or cfg["name"],
        "split": split, "model_path": path, "adapter_dir": adapter_dir,
        "prompt_version": PROMPT_VERSION, "temperature": ev["temperature"],
        "seconds": round(time.time() - started, 1), "environment": environment_info(stack),
        "git_commit": git_commit(repo_root),
        "samples": [{"id": e.task.id, "completion": c} for e, c in
                    list(zip(examples, completions, strict=True))[:20]],
    })
    _write_json(Path(out_path), result)
    return result
