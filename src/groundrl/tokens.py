"""Token-level primitives behind the verifier: claim tokens, sentence splitting, normalisation.

Clean reimplementation of the grounding primitives from the author's `ethresearch-rag`
(MIT licence, same author). The rules differ in several places where this project needs a stricter
reward: ASCII-only digit matching after NFKC normalisation, explicit rejection of invisible or
look-alike characters, a trailing citation attaches to the preceding sentence, and an empty
sentence never passes.

Claim tokens are the things a verifier can check by presence: numbers (decimal, ``1,000``,
``1_000``), hex values, ``2**n`` expressions and ALL_CAPS identifiers.
"""
from __future__ import annotations

import re
import unicodedata

from .protocol import CITATION

_DEC = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:_\d+)*)(?:\.\d+)?"
_NUM = re.compile(
    rf"0[xX][0-9a-fA-F]+(?:_[0-9a-fA-F]+)*|{_DEC}(?:\s?(?:\*\*|\^)\s?\d+)?(?:[eE][+-]?\d+)?",
    re.ASCII,
)
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z][A-Z0-9]{2,}\b", re.ASCII)
STOP_CAPS = frozenset({
    "EIP", "ETH", "EVM", "THE", "AND", "NOT", "MUST", "SHALL", "SHOULD", "MAY", "RLP", "ERC",
    "JSON", "RPC", "API", "URL", "NOTE", "YES", "FOR", "REQUIRED", "RECOMMENDED", "OPTIONAL",
    "TBD", "NONE", "TRUE", "FALSE", "HTTP", "HTTPS",
})
_UNIT = re.compile(r"(\d[\d,_]*)\s*(ETH|ether|Gwei|gwei|wei|[kMG])\b", re.ASCII)
_UNIT_SCALE = {"eth": 10**18, "ether": 10**18, "gwei": 10**9, "wei": 1, "k": 10**3, "m": 10**6,
               "g": 10**9}
_NEGATION = re.compile(r"\b(?:not|never|cannot|no longer|neither|nor)\b|\w+n't\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9_]+")
_STEM = re.compile(r"(ies|ied|ing|ed|es|s)$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+")
_FENCE_LINE = re.compile(r"^\s*(?:```|~~~)[\w+-]*\s*$")
_CITATION_ONLY = re.compile(r"^[\s.,;:!?()\[\]*_`]*(?:\[eip-\d+#\d{3,}\][\s.,;:!?()*_`]*)+$")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\[(\"'*_])")

STOPWORDS = frozenset(
    (
        "a an the of to in on at by for with from and or but not no is are was were be been being "
        "it its this that these those as if then than so such into over under per via can could "
        "will would should shall may might must do does did has have had which who whom whose "
        "what when where how why also only any all each both other more most less least there "
        "their they them we our you your i he she his her"
    ).split()
)

# Non-ASCII characters that are ordinary typography and may legitimately appear in an answer.
_ALLOWED_NON_ASCII = frozenset(
    "–—‘’“”…×≥≤≈≠±·•"
    "→←↔ °μ"
)


def normalize_text(text: str) -> str:
    """NFKC-normalise: fullwidth and superscript digits, ligatures and similar fold to ASCII."""
    return unicodedata.normalize("NFKC", text)


def suspicious_chars(text: str) -> list[str]:
    """Characters in an *answer* that could disguise a claim from the tokenizer.

    Flags invisible/format characters (zero-width joiners, bidi controls), non-ASCII decimal
    digits, and non-ASCII letters (Cyrillic or Greek look-alikes of Latin letters). Ordinary
    typography (curly quotes, dashes, ``>=`` style symbols) is allowed. Control characters other
    than tab/newline/carriage return are flagged too.
    """
    found: list[str] = []
    for ch in text:
        if ch.isascii():
            if ord(ch) < 32 and ch not in "\t\n\r":
                found.append(f"U+{ord(ch):04X}")
            continue
        if ch in _ALLOWED_NON_ASCII:
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N", "C", "M") or cat == "Zs" or ch in "‐‑‒［］":
            found.append(f"U+{ord(ch):04X}")
    return sorted(set(found))


def strip_citations(text: str) -> str:
    return CITATION.sub("", text)


def norm_number(tok: str) -> str:
    """Canonical numeric token: no separators, lower-case, ``0x0B`` -> ``0xb``, ``^`` -> ``**``."""
    t = re.sub(r"[\s,_]", "", tok).lower().rstrip(".").replace("^", "**")
    if t.startswith("0x") and len(t) > 2:
        return "0x" + (t[2:].lstrip("0") or "0")
    return t


def number_variants(tok: str) -> set[str]:
    """Equivalent spellings of a normalised numeric token (hex <-> decimal, ``2**n`` <-> value)."""
    out = {tok}
    try:
        if tok.startswith("0x"):
            out.add(str(int(tok, 16)))
        elif "**" in tok:
            base, exp = tok.split("**", 1)
            if base.isdigit() and exp.isdigit() and int(exp) <= 512:
                out.add(str(int(base) ** int(exp)))
        elif tok.isdigit():
            v = int(tok)
            if v >= 16:
                out.add(norm_number(hex(v)))
            if v > 1 and v & (v - 1) == 0:
                out.add(f"2**{v.bit_length() - 1}")
    except ValueError:  # pragma: no cover - guarded by the isdigit/startswith checks
        pass
    return out


def unit_variants(text: str) -> dict[str, set[str]]:
    """``{"32": {"32000000000000000000", "32000000000"}}`` for ``32 ETH`` style phrases."""
    out: dict[str, set[str]] = {}
    for num, unit in _UNIT.findall(normalize_text(text)):
        base = norm_number(num)
        if not base.isdigit():
            continue
        scale = _UNIT_SCALE[unit.lower()]
        alts = {str(int(base) * scale)}
        if unit.lower() in ("eth", "ether"):
            alts.add(str(int(base) * 10**9))
        out.setdefault(base, set()).update(alts)
    return out


def numeric_tokens(text: str) -> set[str]:
    return {norm_number(m) for m in _NUM.findall(normalize_text(strip_citations(text)))}


def identifier_tokens(text: str) -> set[str]:
    text = normalize_text(strip_citations(text))
    return {m.lower() for m in _IDENT.findall(text) if m not in STOP_CAPS}


def claim_tokens(text: str) -> set[str]:
    """All checkable claim tokens in ``text`` (citations excluded)."""
    return {t for t in numeric_tokens(text) | identifier_tokens(text) if t}


def is_numeric_token(tok: str) -> bool:
    return bool(tok) and (tok[0].isdigit())


def sentences(text: str) -> list[str]:
    """Split an answer into checkable sentences.

    Lines are split first (bullets, table rows and fence markers handled), then each line on
    sentence boundaries. A fragment made only of citations (``... is 4096. [eip-4844#003]``)
    attaches to the preceding sentence. Fence marker lines are dropped but their *content* is
    kept and verified, so code fences cannot hide claims.
    """
    out: list[str] = []
    for raw_line in normalize_text(text).splitlines():
        if _FENCE_LINE.match(raw_line) or not raw_line.strip():
            continue
        line = _LIST_MARKER.sub("", raw_line.strip())
        for part in _BOUNDARY.split(line):
            part = part.strip()
            if not part:
                continue
            if out and _CITATION_ONLY.match(part):
                out[-1] = f"{out[-1]} {part}"
            else:
                out.append(part)
    return out


def stem(tok: str) -> str:
    return _STEM.sub("", tok) if len(tok) > 4 else tok


def content_words(text: str) -> set[str]:
    text = normalize_text(strip_citations(text)).lower()
    return {stem(t) for t in _WORD.findall(text) if t not in STOPWORDS and len(t) > 1}


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(strip_citations(text)))


def word_count(text: str) -> int:
    return len(strip_citations(text).split())
