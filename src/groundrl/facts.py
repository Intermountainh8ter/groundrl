"""Deterministic extraction of checkable facts from chunks.

A *fact* is something a question can be asked about and a verifier can check by token presence:

* ``constant``   a named protocol constant, from a constants table or an assignment line
* ``opcode``     an EVM opcode mnemonic and its byte
* ``type_byte``  a transaction/request type byte
* ``metadata``   a preamble field (``created`` date, ``requires`` list)
* ``cloze``      a prose sentence with one number blanked out

Every fact records the chunk it was read from and the normalised claim tokens of its value
(``tokens``). :func:`supports` decides whether *any* chunk supports a fact, which is how the task
generator guarantees that answerable questions have a supporting chunk in context and that
unanswerable ones do not.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .chunker import Chunk
from .tokens import (
    claim_tokens,
    content_words,
    norm_number,
    normalize_text,
    number_variants,
    numeric_tokens,
    sentences,
)

CONSTANT, OPCODE, TYPE_BYTE = "constant", "opcode", "type_byte"
METADATA, CLOZE = "metadata", "cloze"
KINDS = (CONSTANT, OPCODE, TYPE_BYTE, METADATA, CLOZE)

_NAME = r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+"
_VAL = r"(?:0[xX][0-9a-fA-F]+|\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:_\d+)*(?:\s?\*\*\s?\d+)?)"
_WRAPPER = re.compile(
    r"^(?:Bytes\d*|uint\d*|Address|DomainType|Version|Hash32|Root|ForkDigest|Gwei)"
    r"\(\s*(.*?)\s*\)$"
)
_BARE = re.compile(rf"^[bB]?['\"]?({_VAL})['\"]?$")
_ASSIGN = re.compile(
    rf"^\s*(?:[-*]\s*)?`?(?P<name>{_NAME})`?\s*(?::=|=)\s*`?(?P<val>[^`#\n]+?)`?\s*(?:#.*|\(.*\))?[.,;]?\s*$"
)
_PROSE_DEF = re.compile(
    rf"`(?P<name>{_NAME})`\s*(?:\bis\b|\bset to\b|\bequals\b|:|=)\s*`?(?P<val>{_VAL})`?(?![\w.*])"
    rf"|`(?P<name2>{_NAME})\s*=\s*(?P<val2>{_VAL})`"
)
_OPCODE = re.compile(
    r"`?(?P<name>[A-Z][A-Z0-9]{2,})`?\s*(?:[\s`(:=,]|\b(?:at|opcode|instruction|is|introduced|with|"
    r"value|byte|number|shall|be|created|assigned|will|located|set|to)\b){0,8}?"
    r"`?\(?\s*`?(?P<val>0[xX][0-9a-fA-F]{1,2})(?![0-9a-zA-Z])"
)
_OPCODE_NOISE = frozenset(
    {"EIP", "ERC", "RLP", "ABI", "EVM", "ETH", "MUST", "SHALL", "SHOULD", "NOT"}
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CLOZE_NUM = re.compile(
    r"(?<![\w.#/*-])(0[xX][0-9a-fA-F]+|\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\*\*\d+)?)(?![\w*]|-\d|\.\d)"
)
_CLOZE_BAD_PREFIX = re.compile(
    r"(?:EIP|ERC|RFC|BIP|ISO|SLIP|SEC|FIPS|NIST|CVE|Python|version|Version|v|section|Section|"
    r"Table|Figure|PR|issue)[-\s#]*$"
)
_LIST_TAIL = re.compile(r"(?<=[.!?:])\s+\d{1,2}\.$")
_ITEM_START = re.compile(r"^\s*(?:[-*+]\s|\d{1,2}[.)]\s)")
BLANK = "_____"


@dataclass(frozen=True)
class Fact:
    id: str  # stable and unique: "<chunk id>:<kind>:<key>"
    doc: str
    chunk_id: str
    kind: str
    name: str  # identifier / mnemonic / metadata field; "" for cloze
    value: str  # the value as written in the source
    tokens: tuple[str, ...]  # normalised claim tokens of the value (the gold tokens)
    blanked: str = ""  # cloze only: the sentence with the value replaced by BLANK
    sentence: str = ""  # cloze only: the original sentence (markdown-light)
    section: str = ""

    @property
    def doc_number(self) -> int:
        return int(self.doc.split("-", 1)[1])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = list(self.tokens)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Fact:
        return cls(**{**d, "tokens": tuple(d["tokens"])})


def parse_value(cell: str) -> str | None:
    """A single numeric literal (optionally inside ``Bytes1(0x03)`` style wrappers), else ``None``.

    A trailing gloss such as ``(= 16,384)`` is ignored: it restates the same value.
    """
    cell = re.sub(r"`?\s*\(=[^)]*\)\s*$", "", cell.strip())
    cell = cell.strip().strip("`*_ ").strip()
    wrapped = _WRAPPER.match(cell)
    m = _BARE.match(wrapped.group(1) if wrapped else cell)
    return m.group(1) if m else None


def _classify(name: str, value: str) -> str:
    is_hex = value.lower().startswith("0x")
    if is_hex and "OPCODE" in name:
        return OPCODE
    if is_hex and (name.endswith("_TYPE") or name == "MAGIC"):
        return TYPE_BYTE
    return CONSTANT


def _value_tokens(value: str) -> tuple[str, ...]:
    return tuple(sorted(numeric_tokens(value)))


def _named_candidates(chunk: Chunk) -> Iterable[tuple[str, str]]:
    """``(name, value)`` pairs found in one chunk's body."""
    for line in chunk.body.splitlines():
        if line.count("|") >= 2:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            first = cells[0].strip("`*_ ").strip() if cells else ""
            if len(cells) >= 2 and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", first) and "_" in first:
                val = parse_value(cells[1])
                if val is None and len(cells) >= 3 and re.fullmatch(r"`?[A-Za-z0-9_]+`?", cells[1]):
                    val = parse_value(cells[2])
                if val is not None:
                    yield first, val
            continue
        m = _ASSIGN.match(line)
        if m:
            val = parse_value(m.group("val"))
            if val is not None:
                yield m.group("name"), val
            continue
        for m in _PROSE_DEF.finditer(line):
            yield (m.group("name") or m.group("name2")), (m.group("val") or m.group("val2"))


def _opcode_candidates(chunk: Chunk) -> Iterable[tuple[str, str]]:
    if not re.search(r"opcode|instruction|\bEVM\b", chunk.text, re.IGNORECASE):
        return
    for line in chunk.body.splitlines():
        for m in _OPCODE.finditer(line):
            name = m.group("name")
            if name not in _OPCODE_NOISE:
                yield name, m.group("val")


def _metadata_facts(chunk: Chunk) -> list[Fact]:
    out: list[Fact] = []
    fields = dict(
        line.split(": ", 1) for line in chunk.body.splitlines() if ": " in line
    )
    created = fields.get("created", "")
    if _DATE.match(created):
        out.append(Fact(f"{chunk.id}:metadata:created", chunk.doc, chunk.id, METADATA, "created",
                        created, _value_tokens(created), section=chunk.section))
    req = fields.get("requires", "")
    nums = re.findall(r"\d+", req)
    if nums and re.fullmatch(r"[\d,\s]+", req):
        out.append(Fact(f"{chunk.id}:metadata:requires", chunk.doc, chunk.id, METADATA, "requires",
                        ", ".join(nums), _value_tokens(", ".join(nums)), section=chunk.section))
    return out


def plain(sentence: str) -> str:
    """Strip light markdown (backticks, bold) without touching ``2**17`` exponents."""
    s = sentence.replace("`", "")
    s = re.sub(r"(?<![\w*])\*\*(?=[^\s*])(.+?)(?<=[^\s*])\*\*(?![\w*])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def paragraph_sentences(body: str) -> list[str]:
    """Sentences of ``body`` with hard-wrapped lines re-joined into their paragraph first."""
    items: list[str] = []
    cur: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            if cur:
                items.append(" ".join(cur))
                cur = []
        elif _ITEM_START.match(line) and cur:
            items.append(" ".join(cur))
            cur = [line.strip()]
        else:
            cur.append(line.strip())
    if cur:
        items.append(" ".join(cur))
    return [s for item in items for s in sentences(item)]


def cloze_candidates(chunk: Chunk) -> list[tuple[str, str, str]]:
    """``(sentence, blanked, value)`` for prose sentences, one candidate per blankable number."""
    if chunk.kind != "prose" or chunk.section == "Metadata":
        return []
    out: list[tuple[str, str, str]] = []
    for sent in paragraph_sentences(chunk.body):
        s = plain(_LIST_TAIL.sub("", sent))
        words = len(s.split())
        if not 8 <= words <= 45 or not s[0].isupper() or "http" in s or "|" in s or "](" in s:
            continue
        if "..." in s or "\u2026" in s:
            continue
        if len(numeric_tokens(s)) > 4 or len(claim_tokens(s) - numeric_tokens(s)) > 4:
            continue
        spans = [m for m in _CLOZE_NUM.finditer(s) if not _CLOZE_BAD_PREFIX.search(s[:m.start()])]
        for m in spans[:3]:
            if _in_power_expression(s, m):
                continue
            value = m.group(1)
            if value.isdigit() and len(value) == 1 and (len(s) + ord(value)) % 4:  # keep ~1 in 4
                continue
            if _DATE.match(value) or _is_year(value) or _is_placeholder_hex(value):
                continue
            out.append((s, s[:m.start()] + BLANK + s[m.end():], value))
    return out


def _in_power_expression(s: str, m: re.Match[str]) -> bool:
    """A number that is half of ``2^128`` / ``2 ** 256`` cannot be blanked on its own."""
    return bool(re.search(r"(?:\^|\*\*)\s*$", s[:m.start()])
                or re.match(r"\s*(?:\^|\*\*)", s[m.end():]))


def _is_year(value: str) -> bool:
    return value.isdigit() and len(value) == 4 and 1990 <= int(value) <= 2035


def _is_placeholder_hex(value: str) -> bool:
    body = value[2:].lower() if value.lower().startswith("0x") else ""
    return len(body) >= 3 and len(set(body)) == 1


def extract_facts(chunks: Sequence[Chunk]) -> list[Fact]:
    """All facts in ``chunks``, ambiguity-filtered and in a deterministic order."""
    named: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for c in chunks:
        for name, val in _named_candidates(c):
            named[(c.doc, name)].append((c.id, val))
        for name, val in _opcode_candidates(c):
            named[(c.doc, name)].append((c.id, val))
    facts: list[Fact] = []
    for (doc, name), hits in named.items():
        if len({norm_number(v) for _, v in hits}) != 1:
            continue  # same name, different values: ambiguous, never ask about it
        chunk_id, val = hits[0]
        kind = OPCODE if _is_opcode_name(name, hits) else _classify(name, val)
        facts.append(Fact(f"{chunk_id}:{kind}:{name}", doc, chunk_id, kind, name, val,
                          _value_tokens(val), section=_section_of(chunks, chunk_id)))
    for c in chunks:
        if c.section == "Metadata":
            facts.extend(_metadata_facts(c))
            continue
        seen_blanked: dict[str, int] = defaultdict(int)
        cands = cloze_candidates(c)
        for _, blanked, _ in cands:
            seen_blanked[blanked] += 1
        for i, (sent, blanked, val) in enumerate(cands):
            if seen_blanked[blanked] > 1:
                continue
            facts.append(Fact(f"{c.id}:cloze:{i}", c.doc, c.id, CLOZE, "", val, _value_tokens(val),
                              blanked=blanked, sentence=sent, section=c.section))
    unique = {f.id: f for f in facts}
    return sorted(unique.values(), key=lambda f: (f.doc_number, f.chunk_id, f.id))


def _is_opcode_name(name: str, hits: list[tuple[str, str]]) -> bool:
    # Mnemonics have no underscore (TLOAD, PUSH0) and a one-byte hex value.
    return "_" not in name and all(len(norm_number(v)) <= 4 and v.lower().startswith("0x")
                                   for _, v in hits)


def _section_of(chunks: Sequence[Chunk], chunk_id: str) -> str:
    for c in chunks:
        if c.id == chunk_id:
            return c.section
    return ""  # pragma: no cover - chunk ids always come from ``chunks``


def token_supported(tok: str, chunk_tokens: set[str]) -> bool:
    """True if ``tok`` or an equivalent spelling (hex/decimal/2**n) is among ``chunk_tokens``."""
    return bool(number_variants(tok) & chunk_tokens) if tok[:1].isdigit() else tok in chunk_tokens


def supports(chunk: Chunk, fact: Fact, chunk_tokens: set[str] | None = None) -> bool:
    """Does ``chunk`` contain what is needed to answer a question about ``fact``?"""
    toks = chunk_tokens if chunk_tokens is not None else claim_tokens(chunk.text)
    if not all(token_supported(t, toks) for t in fact.tokens):
        return False
    if fact.kind == METADATA:
        return chunk.doc == fact.doc and chunk.section == "Metadata"
    if fact.kind == CLOZE:
        want = content_words(normalize_text(fact.blanked.replace(BLANK, " ")))
        if not want:
            return False  # pragma: no cover - cloze sentences always have content words
        return len(want & content_words(chunk.text)) / len(want) >= 0.7
    return fact.name.lower() in toks
