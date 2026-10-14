"""A fake training stack for testing trainer plumbing without torch, TRL or a GPU.

**Everything produced through this module is MOCK.** The fake trainers do not learn anything; the
fake model answers with whatever scripted policy it is given. Runs through it set ``"mock": true``
in their result JSON, and ``scripts/make_report.py`` never counts such files as results. Its only
purpose is to prove that configs, datasets, reward wrappers, the evaluation pipeline and the file
outputs are wired together correctly before anyone rents a GPU.

``strict_fields`` optionally maps config class names to the set of accepted keyword names, so a
test can make the fake ``SFTConfig`` / ``GRPOConfig`` / ``LoraConfig`` reject exactly the keywords
the real classes would reject.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .prompts import render_chat_stub
from .trainplan import StubTokenizer


class FakeArray(np.ndarray):
    """numpy array with the ``.to(device)`` method generation code calls on torch tensors."""

    def to(self, device: Any) -> FakeArray:
        return self


def _arr(data: Any) -> FakeArray:
    return np.asarray(data, dtype=np.int64).view(FakeArray)


class _Registry:
    """Maps text to a stable integer id and back, standing in for a tokenizer vocabulary."""

    def __init__(self) -> None:
        self.by_id: dict[int, str] = {}

    def encode(self, text: str) -> int:
        i = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:6], "big") + 10
        self.by_id[i] = text
        return i

    def decode(self, i: int) -> str:
        return self.by_id.get(int(i), "")


class MockStack(SimpleNamespace):
    """Build with :func:`make_mock_stack`; this class only exists for typing and repr."""


def make_mock_stack(
    policy: Callable[[str], str], strict_fields: Mapping[str, set[str]] | None = None,
    record: list[tuple[str, dict[str, Any]]] | None = None,
) -> MockStack:
    """A fake stack whose model answers ``policy(prompt_text)``.

    ``record`` (if given) receives ``(event, payload)`` tuples for every constructor call, so a
    test can assert on exactly what the runners passed to each library class.
    """
    registry = _Registry()
    stub = StubTokenizer()
    log = record if record is not None else []
    strict = dict(strict_fields or {})

    def check(cls_name: str, kwargs: Mapping[str, Any]) -> None:
        allowed = strict.get(cls_name)
        if allowed is not None:
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"{cls_name} got unexpected keyword argument(s) {sorted(unknown)}")

    class Torch:
        __version__ = "mock"
        bfloat16, float16, float32 = "bfloat16", "float16", "float32"

        class cuda:  # noqa: N801 - mirrors torch.cuda
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def no_grad() -> contextlib.AbstractContextManager:
            return contextlib.nullcontext()

    class Tokenizer:
        pad_token = None
        eos_token = "<|im_end|>"
        pad_token_id = 0
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, path: str, **kw: Any) -> Tokenizer:
            log.append(("tokenizer", {"path": path, **kw}))
            return cls()

        def apply_chat_template(self, messages: Any, tokenize: bool = False,
                                add_generation_prompt: bool = True,
                                enable_thinking: bool = True) -> str:
            assert not tokenize
            return render_chat_stub(messages, add_generation_prompt, enable_thinking)

        def __call__(self, text: Any, return_tensors: str | None = None, padding: bool = False,
                     add_special_tokens: bool = True) -> dict[str, Any]:
            texts = [text] if isinstance(text, str) else list(text)
            if return_tensors is None:
                ids = [[1] * stub.count(t) for t in texts]
                return {"input_ids": ids[0] if isinstance(text, str) else ids}
            width = 4
            rows = [[registry.encode(t)] + [0] * (width - 1) for t in texts]
            return {"input_ids": _arr(rows), "attention_mask": _arr([[1] * width] * len(rows))}

        def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
            return [registry.decode(row[0]) if len(row) else "" for row in np.asarray(ids)]

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text('{"mock": true}')

    class Model:
        device = "cpu"

        def __init__(self, path: str = "", **kw: Any) -> None:
            self.path = path
            self.kwargs = kw

        @classmethod
        def from_pretrained(cls, path: str, **kw: Any) -> Model:
            log.append(("model", {"path": path, **kw}))
            return cls(path, **kw)

        def eval(self) -> Model:
            return self

        def generate(self, input_ids: Any, attention_mask: Any = None, **kw: Any) -> FakeArray:
            log.append(("generate", dict(kw)))
            rows = []
            for row in np.asarray(input_ids):
                prompt = registry.decode(row[0])
                rows.append(list(row) + [registry.encode(policy(prompt))])
            return _arr(rows)

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text('{"mock": true}')

    class PeftModel(Model):
        @classmethod
        def from_pretrained(cls, base: Model, path: str, **kw: Any) -> PeftModel:  # type: ignore[override]
            log.append(("peft_model", {"path": path}))
            inst = cls(base.path)
            return inst

        def merge_and_unload(self) -> Model:
            return Model(self.path)

    class Config:
        _name = "Config"

        def __init__(self, **kwargs: Any) -> None:
            check(self._name, kwargs)
            self.kwargs = kwargs
            log.append((self._name, dict(kwargs)))

    class SFTConfig(Config):
        _name = "SFTConfig"

    class GRPOConfig(Config):
        _name = "GRPOConfig"

    class LoraConfig(Config):
        _name = "LoraConfig"

    class Dataset(list):
        @classmethod
        def from_list(cls, rows: list[dict[str, Any]]) -> Dataset:
            return cls(rows)

        @property
        def column_names(self) -> list[str]:
            return list(self[0]) if self else []

    class _Trainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.args = kwargs["args"]
            self.state = SimpleNamespace(log_history=[])
            log.append((type(self).__name__, {k: type(v).__name__ for k, v in kwargs.items()}))

        def save_model(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text(json.dumps({"mock": True}))

    class SFTTrainer(_Trainer):
        def train(self) -> None:
            n = len(self.kwargs["train_dataset"])
            self.state.log_history = [{"step": 1, "mock": 1, "examples": n}]
            assert self.kwargs["train_dataset"][0]["completion"], "SFT rows need a completion"

    class GRPOTrainer(_Trainer):
        def train(self) -> None:
            ds = self.kwargs["train_dataset"]
            reward = self.kwargs["reward_funcs"]
            rows = ds[: max(2, getattr(self.args, "kwargs", {}).get("num_generations", 2))]
            comps = [policy(r["prompt"]) for r in rows]
            columns = {k: [r[k] for r in rows] for k in rows[0] if k != "prompt"}
            scores = reward(prompts=[r["prompt"] for r in rows], completions=comps, **columns)
            assert len(scores) == len(rows)
            self.state.log_history = [{"step": 1, "mock": 1,
                                       "mock_scripted_reward_mean": float(np.mean(scores))}]

    return MockStack(
        torch=Torch, transformers=SimpleNamespace(__version__="mock"),
        trl=SimpleNamespace(__version__="mock"), peft=SimpleNamespace(__version__="mock"),
        datasets=SimpleNamespace(__version__="mock"),
        AutoModelForCausalLM=Model, AutoTokenizer=Tokenizer,
        BitsAndBytesConfig=lambda **kw: SimpleNamespace(**kw), set_seed=lambda s: None,
        LoraConfig=LoraConfig, PeftModel=PeftModel, SFTConfig=SFTConfig, SFTTrainer=SFTTrainer,
        GRPOConfig=GRPOConfig, GRPOTrainer=GRPOTrainer, Dataset=Dataset, log=log,
    )
