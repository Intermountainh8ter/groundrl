"""The answer protocol shared by prompts, verifier and rewards.

Keeping these constants in one dependency-free module means a change to the citation syntax or the
abstention sentence cannot silently drift between the prompt a model sees and the check that
scores it.
"""
from __future__ import annotations

import re

# A citation names a chunk id: lower-case ``eip-<number>#<zero padded chunk index>``.
CHUNK_ID = re.compile(r"eip-\d+#\d{3,}")
CITATION = re.compile(r"\[(eip-\d+#\d{3,})\]")
# Anything that *looks* like an attempted citation. Used only to diagnose format drift.
LOOSE_CITATION = re.compile(
    r"[\[(（［]\s*(?:eip|erc)[-‐-―\s_]*\d+\s*[#:]\s*\d+\s*[\])）］]", re.IGNORECASE
)

ABSTAIN_TEXT = "I cannot answer this from the provided context."
_ABSTAIN_KEY = re.compile(r"[\W_]+")


def abstain_key(text: str) -> str:
    """Canonical comparison form of a would-be abstention: case-folded, punctuation-free."""
    return _ABSTAIN_KEY.sub(" ", text).strip().lower()


_ABSTAIN_CANON = abstain_key(ABSTAIN_TEXT)


def is_abstention(text: str) -> bool:
    """True only if the *entire* completion is the canonical abstention sentence.

    The match is strict on purpose: "abstain, then answer anyway" must not count as abstaining.
    Case, surrounding whitespace and trailing punctuation are ignored.
    """
    return abstain_key(text) == _ABSTAIN_CANON
