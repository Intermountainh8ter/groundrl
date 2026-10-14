"""Synthetic EIP-like documents for fast, offline tests."""
from __future__ import annotations

from groundrl.chunker import chunk_corpus


def make_doc(n: int) -> str:
    """A small proposal with a constants table, an assignment, an opcode line and prose."""
    return f"""---
eip: {n}
title: Synthetic proposal {n}
status: Final
type: Standards Track
category: Core
created: 2021-0{n % 9 + 1}-1{n % 8}
requires: {n - 1}, {n - 2}
---

## Abstract
This proposal sets the widget limit to {n * 3 + 11} units for every account on the network today.
It also changes how the gas schedule treats widgets in a way that is easy to audit later on.

## Specification
The following constants are used.

| Constant | Value |
| - | - |
| `LIMIT_{n}_A` | `{n * 100 + 7}` |
| `LIMIT_{n}_B` | `0x{n % 200 + 16:02x}` |
| `LIMIT_{n}_C` | `2**{n % 20 + 3}` |

A new opcode, `WIDGET{n}`, is introduced at `0x{n % 90 + 10:02x}`, with a gas cost of 3.

```python
FORK_BLOCK_{n} = {n * 1000}
```

## Rationale
The cost of {n * 7 + 5} gas per widget was chosen so that the average block stays well within the
existing limits. Longer discussion follows here about why prose chunks should be split on
sentence boundaries and never in the middle of a sentence that carries a number.

## Copyright
Copyright and related rights waived via CC0.
"""


def make_docs(numbers: range | list[int]) -> dict[str, str]:
    return {f"eip-{n}": make_doc(n) for n in numbers}


def make_chunks(numbers: range | list[int]):
    return chunk_corpus(make_docs(numbers))
