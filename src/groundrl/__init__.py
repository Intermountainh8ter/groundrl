"""groundrl: a deterministic citation-grounding verifier used as a verifiable reward for RL.

The core library depends only on the standard library and numpy. Heavy training dependencies
(torch, transformers, trl, peft, datasets, vllm) are optional extras that are imported lazily by
the scripts under ``train/`` and never by this package.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
