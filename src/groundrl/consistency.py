"""Detect an answer that says the same thing twice with different numbers (hedging).

Two sentences say the same thing if their words are identical once pure numbers are removed
(EIP numbers are kept, so "EIP-198 was created on ..." and "EIP-1167 was created on ..." differ).
They contradict if both carry numeric claims and the sets differ. Legitimate multi-fact answers
use different words for different facts, so they do not trigger this. A hedge that rewords the
sentence is not caught here; the extra-claim penalty in the reward still applies to it.

The task generator uses the same function to refuse any question whose own gold answer would
count as contradicting itself, so the reward can never punish the oracle.
"""
from __future__ import annotations

from collections.abc import Collection

from .tokens import claim_tokens, content_words, number_variants, sentences, strip_citations


def canonical_number(tok: str) -> str:
    """Decimal spelling of a numeric token when one exists (0x40 -> 64), else the token."""
    dec = [v for v in number_variants(tok) if v.isdigit()]
    return dec[0] if dec else tok


def word_signature(text: str, keep: Collection[str] = ()) -> frozenset[str]:
    """Content words of ``text`` that are not numbers (words in ``keep`` are retained)."""
    return frozenset(w for w in content_words(strip_citations(text))
                     if not w[:1].isdigit() or w in keep)


def count_contradictions(answer: str, eip_numbers: Collection[str]) -> int:
    """Number of sentence pairs with identical words and different numeric claims."""
    eips = set(eip_numbers)
    rows: list[tuple[frozenset[str], set[str]]] = []
    for s in sentences(answer):
        body = strip_citations(s)
        nums = {canonical_number(t) for t in claim_tokens(body) if t[:1].isdigit()} - eips
        rows.append((word_signature(body, eips), nums))
    count = 0
    for i, (wi, ni) in enumerate(rows):
        for wj, nj in rows[i + 1:]:
            if wi and wi == wj and ni and nj and ni != nj:
                count += 1
    return count
