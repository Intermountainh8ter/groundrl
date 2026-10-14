"""Deterministic grounding verifier with per-claim provenance.

Adapted from ``ethrag.verify`` in the author's `ethresearch-rag` repository (MIT licence, same
author); this is a clean reimplementation with stricter rules for use as a reward signal.

An answer is accepted only if all of the following hold:

1. it is not empty and every sentence has content besides its citations;
2. every sentence carries at least one citation of the form ``[eip-N#CCC]``;
3. every citation names a chunk that was in the provided context (anything else is spoofed or
   hallucinated);
4. every claim token in a sentence (number, hex value, ``2**n`` expression, ALL_CAPS identifier)
   appears as a token in the chunks *that sentence cites*, verbatim or as a recorded equivalent
   (``0x48`` = ``72``, ``2**12`` = ``4096``, ``32 ETH`` = ``32000000000`` gwei);
5. the answer contains no invisible, look-alike or non-ASCII digit/letter characters that could
   hide a claim from the tokenizer.

Substring matches never count (``8`` is not found inside ``58``). Each supported token records
which chunk supported it and how (``exact``, ``hex-dec``, ``pow2``, ``unit``).

What the verifier cannot do: it checks token presence, not entailment. A paraphrase with no
claim token passes vacuously and a correct token placed in a wrong claim passes. The reward
revisions in :mod:`groundrl.rewards` exist because of those two holes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .protocol import CITATION, LOOSE_CITATION
from .tokens import (
    claim_tokens,
    content_words,
    number_variants,
    sentences,
    strip_citations,
    suspicious_chars,
    unit_variants,
)


@dataclass
class ClaimCheck:
    sentence: int
    token: str
    supported: bool
    chunk: str = ""
    kind: str = ""  # exact | hex-dec | pow2 | unit


@dataclass
class SentenceReport:
    index: int
    text: str
    citations: list[str]  # every well-formed citation, in order, duplicates kept
    valid: list[str]  # distinct citations that name a chunk in the context
    invalid: list[str]  # distinct citations that do not
    checked: int = 0
    supported: int = 0
    coverage: float = 1.0
    has_content: bool = True

    @property
    def support_ratio(self) -> float:
        return 1.0 if self.checked == 0 else self.supported / self.checked

    @property
    def passed(self) -> bool:
        return (self.has_content and bool(self.valid) and not self.invalid
                and self.supported == self.checked)


@dataclass
class VerificationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    sentences: list[SentenceReport] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    suspicious: list[str] = field(default_factory=list)
    near_miss: int = 0  # citation-shaped strings that are not well-formed citations

    @property
    def checked_tokens(self) -> int:
        return len(self.claims)

    @property
    def supported_tokens(self) -> int:
        return sum(c.supported for c in self.claims)

    @property
    def support_ratio(self) -> float:
        return 1.0 if not self.claims else self.supported_tokens / self.checked_tokens

    @property
    def cited_chunks(self) -> list[str]:
        """Distinct valid citations over the whole answer, in order of first use."""
        out: list[str] = []
        for s in self.sentences:
            for c in s.valid:
                if c not in out:
                    out.append(c)
        return out

    @property
    def invalid_citations(self) -> list[str]:
        out: list[str] = []
        for s in self.sentences:
            for c in s.invalid:
                if c not in out:
                    out.append(c)
        return out

    def provenance(self) -> dict[str, list[str]]:
        """``{chunk_id: [tokens it supported]}``."""
        out: dict[str, list[str]] = {}
        for c in self.claims:
            if c.supported:
                out.setdefault(c.chunk, []).append(c.token)
        return out

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": self.issues,
            "checked_tokens": self.checked_tokens,
            "supported_tokens": self.supported_tokens,
            "support_ratio": self.support_ratio,
            "cited_chunks": self.cited_chunks,
            "invalid_citations": self.invalid_citations,
            "suspicious": self.suspicious,
            "near_miss": self.near_miss,
            "provenance": self.provenance(),
            "claims": [vars(c) for c in self.claims],
            "sentences": [
                {"index": s.index, "text": s.text, "valid": s.valid, "invalid": s.invalid,
                 "checked": s.checked, "supported": s.supported, "coverage": s.coverage}
                for s in self.sentences
            ],
        }


def _find_support(
    tok: str, cited: Sequence[str], chunk_tokens: Mapping[str, set[str]],
    units: Mapping[str, set[str]],
) -> tuple[str, str]:
    """``(chunk id, kind)`` of the first cited chunk supporting ``tok``; ``("", "")`` if none."""
    for cid in cited:
        if tok in chunk_tokens[cid]:
            return cid, "exact"
    variants = number_variants(tok) - {tok}
    for cid in cited:
        for v in sorted(variants):
            if v in chunk_tokens[cid]:
                return cid, "hex-dec" if (v.startswith("0x") or tok.startswith("0x")) else "pow2"
    for cid in cited:
        for v in sorted(units.get(tok, ())):
            if v in chunk_tokens[cid]:
                return cid, "unit"
    return "", ""


def verify_answer(
    answer: str, chunks: Mapping[str, str], context_ids: Sequence[str] | None = None,
) -> VerificationResult:
    """Verify ``answer`` against ``chunks`` (id -> text). ``context_ids`` defaults to all ids.

    Only ids that are in ``context_ids`` *and* in ``chunks`` count as valid citations.
    """
    res = VerificationResult(ok=True)
    context = set(chunks if context_ids is None else context_ids) & set(chunks)
    res.suspicious = suspicious_chars(answer)
    if res.suspicious:
        res.ok = False
        res.issues.append(f"suspicious characters in answer: {', '.join(res.suspicious)}")
    strict = len(CITATION.findall(answer))
    res.near_miss = max(0, len(LOOSE_CITATION.findall(answer)) - strict)
    if res.near_miss:
        res.issues.append(f"{res.near_miss} malformed citation(s) (expected [eip-N#CCC])")
    sents = sentences(answer)
    if not sents:
        res.ok = False
        res.issues.append("empty answer")
        return res

    token_cache: dict[str, set[str]] = {}
    words_cache: dict[str, set[str]] = {}

    def tokens_of(cid: str) -> set[str]:
        if cid not in token_cache:
            toks = claim_tokens(chunks[cid])
            # "32 ETH" in the chunk also supports the claim "32000000000" (its gwei spelling).
            for alts in unit_variants(chunks[cid]).values():
                toks |= alts
            token_cache[cid] = toks
        return token_cache[cid]

    def words_of(cid: str) -> set[str]:
        if cid not in words_cache:
            words_cache[cid] = content_words(chunks[cid])
        return words_cache[cid]

    for si, sent in enumerate(sents):
        cites = CITATION.findall(sent)
        valid = list(dict.fromkeys(c for c in cites if c in context))
        invalid = list(dict.fromkeys(c for c in cites if c not in context))
        body = strip_citations(sent)
        has_content = any(ch.isalnum() for ch in body)
        report = SentenceReport(si, sent, cites, valid, invalid, has_content=has_content)
        res.sentences.append(report)
        short = sent[:70]
        if not has_content:
            res.ok = False
            res.issues.append(f"sentence {si} has no content besides citations: {short!r}")
        if not cites:
            res.ok = False
            res.issues.append(f"uncited sentence {si}: {short!r}")
        for c in invalid:
            res.ok = False
            res.issues.append(f"citation {c} is not in the provided context")
        tokens_by_chunk = {c: tokens_of(c) for c in valid}
        units = unit_variants(body)
        for tok in sorted(claim_tokens(body)):
            cid, kind = _find_support(tok, valid, tokens_by_chunk, units)
            res.claims.append(ClaimCheck(si, tok, bool(cid), cid, kind))
            report.checked += 1
            if cid:
                report.supported += 1
            else:
                res.ok = False
                res.issues.append(f"unsupported claim {tok!r} in sentence {si} "
                                  f"(cited: {valid or 'none'})")
        words = content_words(body)
        support_words: set[str] = set().union(*(words_of(c) for c in valid)) if valid else set()
        report.coverage = len(words & support_words) / len(words) if words else 1.0
    return res
