"""JSON training configuration: loading, validation and command-line overrides.

A config describes one model at one hardware profile and holds every stage (SFT baseline, GRPO,
evaluation). It is plain JSON (no YAML dependency) and is validated without importing torch:
unknown keys are errors, ranges are checked, and cross-field constraints that TRL would otherwise
reject minutes into a run (batch divisibility, token budgets) are caught up front.

The ``args`` dictionaries of the ``sft`` and ``grpo`` sections are passed verbatim to TRL's
``SFTConfig`` and ``GRPOConfig``; the test suite checks every key against the field names of the
pinned TRL/transformers sources.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SECTIONS = ("name", "description", "profile", "model", "lora", "data", "sft", "grpo", "eval")
DTYPES = ("bfloat16", "float16", "float32")
REWARD_CHOICES = ("v0", "v1", "v2")

_OPT_STR = (str, type(None))
_OPT_INT = (int, type(None))
_MODEL_KEYS = {"name_or_path": str, "dtype": str, "load_in_4bit": bool,
               "attn_implementation": str, "trust_remote_code": bool, "init_from": _OPT_STR}
_LORA_KEYS = {"r": int, "alpha": int, "dropout": float, "target_modules": (str, list),
              "bias": str}
_DATA_KEYS = {"data_dir": str, "train_split": str, "eval_split": str, "max_prompt_tokens": int,
              "max_completion_tokens": int, "enable_thinking": bool, "limit_train": _OPT_INT,
              "limit_eval": _OPT_INT}
_EVAL_KEYS = {"split": str, "batch_size": int, "max_new_tokens": int, "temperature": float,
              "bootstrap": int}
_PROFILE_KEYS = {"vram_gb": (int, float), "notes": str, "estimates_only": bool}
_GRPO_KEYS = {"reward": str, "args": dict}
_SFT_KEYS = {"args": dict}

REQUIRED = {
    "model": {"name_or_path"},
    "lora": {"r", "alpha"},
    "data": {"max_prompt_tokens", "max_completion_tokens"},
    "grpo": {"reward", "args"},
    "sft": {"args"},
}


class ConfigError(ValueError):
    """Raised for an invalid training configuration."""


def _check_keys(section: str, data: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    unknown = set(data) - set(spec)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {sorted(unknown)}; allowed: {sorted(spec)}")
    for key, value in data.items():
        types = spec[key]
        ok = isinstance(value, types)
        if types is int or (isinstance(types, tuple) and int in types):
            ok = ok and not isinstance(value, bool)
        if types is float or (isinstance(types, tuple) and float in types):
            ok = ok or (isinstance(value, int) and not isinstance(value, bool))
        if not ok:
            raise ConfigError(f"{section}.{key}: expected {types}, got {type(value).__name__}")
    missing = REQUIRED.get(section, set()) - set(data)
    if missing:
        raise ConfigError(f"{section}: missing required key(s) {sorted(missing)}")


def validate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated deep copy of ``cfg`` with defaults filled in, or raise ConfigError."""
    cfg = copy.deepcopy(dict(cfg))
    unknown = set(cfg) - set(SECTIONS)
    if unknown:
        raise ConfigError(f"unknown top-level section(s) {sorted(unknown)}")
    for needed in ("name", "model", "lora", "data", "sft", "grpo"):
        if needed not in cfg:
            raise ConfigError(f"missing section {needed!r}")
    if not isinstance(cfg["name"], str) or not cfg["name"]:
        raise ConfigError("name must be a non-empty string")
    cfg.setdefault("description", "")
    cfg["profile"] = {"vram_gb": 24, "notes": "", "estimates_only": True, **cfg.get("profile", {})}
    cfg["model"] = {"dtype": "bfloat16", "load_in_4bit": False, "attn_implementation": "sdpa",
                    "trust_remote_code": False, "init_from": None, **cfg["model"]}
    cfg["lora"] = {"dropout": 0.05, "target_modules": "all-linear", "bias": "none", **cfg["lora"]}
    cfg["data"] = {"data_dir": "data", "train_split": "train", "eval_split": "val",
                   "enable_thinking": False, "limit_train": None, "limit_eval": None,
                   **cfg["data"]}
    cfg["eval"] = {"split": "test", "batch_size": 8, "max_new_tokens": 320, "temperature": 0.0,
                   "bootstrap": 2000, **cfg.get("eval", {})}
    for section, spec in (("profile", _PROFILE_KEYS), ("model", _MODEL_KEYS), ("lora", _LORA_KEYS),
                          ("data", _DATA_KEYS), ("eval", _EVAL_KEYS), ("sft", _SFT_KEYS),
                          ("grpo", _GRPO_KEYS)):
        _check_keys(section, cfg[section], spec)
    _check_values(cfg)
    return cfg


def _check_values(cfg: dict[str, Any]) -> None:
    if cfg["model"]["dtype"] not in DTYPES:
        raise ConfigError(f"model.dtype must be one of {DTYPES}")
    lora = cfg["lora"]
    if lora["r"] < 1 or lora["alpha"] < 1 or not 0.0 <= lora["dropout"] < 1.0:
        raise ConfigError("lora: r and alpha must be >= 1 and dropout in [0, 1)")
    data = cfg["data"]
    if data["max_prompt_tokens"] < 64 or data["max_completion_tokens"] < 16:
        raise ConfigError("data: token budgets are too small to hold a prompt and an answer")
    if cfg["grpo"]["reward"] not in REWARD_CHOICES:
        raise ConfigError(f"grpo.reward must be one of {REWARD_CHOICES}")
    ev = cfg["eval"]
    if ev["temperature"] < 0 or ev["batch_size"] < 1 or ev["max_new_tokens"] < 16:
        raise ConfigError("eval: temperature >= 0, batch_size >= 1, max_new_tokens >= 16")
    total = data["max_prompt_tokens"] + data["max_completion_tokens"]
    sft = cfg["sft"]["args"]
    if "max_length" in sft and sft["max_length"] < total:
        raise ConfigError(f"sft.args.max_length ({sft['max_length']}) is below the prompt plus "
                          f"completion budget ({total})")
    _check_common_args("sft", sft)
    g = cfg["grpo"]["args"]
    _check_common_args("grpo", g)
    if g.get("max_completion_length") != data["max_completion_tokens"]:
        raise ConfigError("grpo.args.max_completion_length must equal data.max_completion_tokens "
                          "so the training and evaluation budgets cannot diverge")
    n = g.get("num_generations")
    if not isinstance(n, int) or n < 2:
        raise ConfigError("grpo.args.num_generations must be an integer >= 2")
    per_device = g.get("per_device_train_batch_size", 8)
    accum = g.get("gradient_accumulation_steps", 1)
    steps_per_gen = g.get("steps_per_generation") or accum
    if (per_device * steps_per_gen) % n != 0:
        raise ConfigError(
            f"per_device_train_batch_size ({per_device}) x steps_per_generation ({steps_per_gen}) "
            f"must be divisible by num_generations ({n}): every generation batch must hold whole "
            "prompt groups")
    if g.get("beta", 0.0) < 0 or g.get("learning_rate", 1.0) <= 0:
        raise ConfigError("grpo.args: beta >= 0 and learning_rate > 0")
    if g.get("remove_unused_columns", False):
        raise ConfigError("grpo.args.remove_unused_columns must be false: the reward needs the "
                          "example_json column")
    if g.get("temperature", 1.0) <= 0:
        raise ConfigError("grpo.args.temperature must be > 0 (sampling is required)")


def _check_common_args(stage: str, args: Mapping[str, Any]) -> None:
    lr = args.get("learning_rate")
    if lr is not None and not 0 < lr < 1:
        raise ConfigError(f"{stage}.args.learning_rate must be in (0, 1)")
    warm = args.get("warmup_steps")
    if warm is not None and (warm < 0 or (isinstance(warm, float) and warm >= 1.0
                                          and warm != int(warm))):
        raise ConfigError(f"{stage}.args.warmup_steps must be an integer >= 0 or a ratio in [0, 1)")
    if "warmup_ratio" in args:
        raise ConfigError(f"{stage}.args.warmup_ratio is not a transformers>=5 argument; use "
                          "warmup_steps with a float ratio")
    if "output_dir" in args:
        raise ConfigError(f"{stage}.args.output_dir is set by the runner (--out); remove it")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if key in args and (not isinstance(args[key], int) or args[key] < 1):
            raise ConfigError(f"{stage}.args.{key} must be an integer >= 1")
    if "seed" in args and not isinstance(args["seed"], int):
        raise ConfigError(f"{stage}.args.seed must be an integer")


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON ({exc})") from exc
    return validate_config(raw)


def set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """``set_path(cfg, "grpo.args.beta", 0.04)``; the leaf must already exist or be new in args."""
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"override path {dotted!r}: no section {part!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise ConfigError(f"override path {dotted!r} does not point into a section")
    node[parts[-1]] = value


def apply_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply dotted-path overrides and re-validate; the input is not modified."""
    out = copy.deepcopy(dict(cfg))
    for dotted, value in overrides.items():
        set_path(out, dotted, value)
    return validate_config(out)


def cli_overrides(
    cfg: Mapping[str, Any], seed: int | None = None, lr: float | None = None,
    group_size: int | None = None, kl_beta: float | None = None, max_len: int | None = None,
    sets: list[str] | None = None, stage: str = "grpo",
) -> dict[str, Any]:
    """Translate the convenience flags and ``--set key=json`` pairs into dotted overrides.

    ``max_len`` is the total token budget (prompt + completion): the completion budget stays and
    the prompt budget and the SFT ``max_length`` follow from it.
    """
    ov: dict[str, Any] = {}
    if seed is not None:
        ov["sft.args.seed"] = seed
        ov["grpo.args.seed"] = seed
    if lr is not None:
        ov[f"{stage}.args.learning_rate"] = lr
    if group_size is not None:
        ov["grpo.args.num_generations"] = group_size
    if kl_beta is not None:
        ov["grpo.args.beta"] = kl_beta
    if max_len is not None:
        completion = cfg["data"]["max_completion_tokens"]
        if max_len <= completion + 64:
            raise ConfigError(f"--max-len {max_len} leaves no room for a prompt after "
                              f"{completion} completion tokens")
        ov["data.max_prompt_tokens"] = max_len - completion
        ov["sft.args.max_length"] = max_len
    for item in sets or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        try:
            ov[key] = json.loads(raw)
        except json.JSONDecodeError:
            ov[key] = raw
    return ov
