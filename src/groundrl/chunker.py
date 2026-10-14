"""Section-, table- and code-aware chunking of EIP markdown into citable chunks.

Design goals, in order: a chunk never cuts a fenced code block or a table row in half; a long
table is split by rows with its header repeated, so every row keeps its column names; prose is
split on sentence boundaries; every chunk carries a one-line context header
(``EIP-1559: <title> / <section>``) so the document identity is part of the citable text; ids are
stable (``eip-1559#007``) so that tasks can refer to chunks by id alone.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .corpus import front_matter

DEFAULT_MAX_WORDS = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$|^[^|\n]+\|[^|\n]*\|.*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\[(\"'*])")
_DROP_SECTIONS = {"copyright", "citation"}
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_EIP_LINK = re.compile(r"\[[^\]]*\]\((?:\.{0,2}/)*(?:eip|erc)-(\d+)(?:\.md)?[^)]*\)", re.IGNORECASE)
_EIP_REF = re.compile(r"\[[^\]]*\]\[(?:eip|erc)-(\d+)\]", re.IGNORECASE)
_PLAIN_REF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    number: int
    title: str
    section: str
    kind: str  # prose | code | table
    text: str  # header line + body; this is what a model sees and what a citation points at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)

    @property
    def body(self) -> str:
        nl = self.text.find("\n")
        return self.text[nl + 1:] if nl >= 0 else self.text


def clean_markdown(body: str) -> str:
    """Drop HTML comments and reduce links to their text (urls only add noise).

    Links to other proposals become ``EIP-N`` whatever their link text was, so a bare ``[2718]``
    cannot be mistaken for a quantity.
    """
    body = _HTML_COMMENT.sub("", body)
    body = _EIP_LINK.sub(lambda m: f"EIP-{m.group(1)}", body)
    body = _EIP_REF.sub(lambda m: f"EIP-{m.group(1)}", body)
    body = _LINK.sub(lambda m: m.group(1), body)
    return _PLAIN_REF.sub(lambda m: m.group(1), body)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings; ``#`` lines inside fenced code are not headings."""
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    in_code = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            sections.append((m.group(2).strip().strip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split section text into atomic ``(kind, text)`` blocks: prose, code, table."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    para: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if para and "\n".join(para).strip():
            blocks.append(("prose", "\n".join(para).strip()))
        para.clear()

    while i < n:
        line = lines[i]
        if _FENCE.match(line):
            flush()
            fence = [line]
            i += 1
            while i < n and not _FENCE.match(lines[i]):
                fence.append(lines[i])
                i += 1
            if i < n:
                fence.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(fence)))
            continue
        if _TABLE_ROW.match(line) and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(rows)))
            continue
        if not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


def _words(text: str) -> int:
    return len(text.split())


def split_prose(text: str, max_words: int) -> list[str]:
    """Split oversized prose on sentence boundaries (never inside a sentence unless forced)."""
    if _words(text) <= max_words:
        return [text]
    pieces: list[str] = []
    cur: list[str] = []
    count = 0
    for sent in _SENT_SPLIT.split(text.replace("\n", " ")):
        w = _words(sent)
        if w > max_words:  # one enormous sentence: hard-wrap on words
            if cur:
                pieces.append(" ".join(cur))
                cur, count = [], 0
            toks = sent.split()
            pieces.extend(" ".join(toks[i:i + max_words]) for i in range(0, len(toks), max_words))
            continue
        if cur and count + w > max_words:
            pieces.append(" ".join(cur))
            cur, count = [], 0
        cur.append(sent)
        count += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_code(text: str, max_words: int) -> list[str]:
    """Window a fenced block by lines, re-wrapping every piece in the original fences."""
    lines = text.splitlines()
    if sum(_words(ln) for ln in lines) <= max_words:
        return [text]
    open_fence = lines[0] if _FENCE.match(lines[0]) else ""
    close_fence = lines[-1] if len(lines) > 1 and _FENCE.match(lines[-1]) else ""
    inner = lines[1 if open_fence else 0: len(lines) - (1 if close_fence else 0)]
    pieces: list[list[str]] = []
    cur: list[str] = []
    count = 0
    for ln in inner:
        for part in (_hard_wrap(ln, max_words) if _words(ln) > max_words else [ln]):
            w = max(1, _words(part))
            if cur and count + w > max_words:
                pieces.append(cur)
                cur, count = [], 0
            cur.append(part)
            count += w
    if cur:
        pieces.append(cur)
    return ["\n".join([open_fence, *p, close_fence] if open_fence else p) for p in pieces]


def _hard_wrap(line: str, max_words: int) -> list[str]:
    toks = line.split()
    return [" ".join(toks[i:i + max_words]) for i in range(0, len(toks), max_words)]


def split_table(text: str, max_words: int) -> list[str]:
    """Split a long table by rows, repeating header and separator on every piece."""
    rows = text.splitlines()
    header, body = rows[:2], rows[2:]
    hw = sum(_words(r) for r in header)
    if hw + sum(_words(r) for r in body) <= max_words:
        return [text]
    pieces: list[str] = []
    cur: list[str] = []
    count = hw
    for r in body:
        w = _words(r)
        if cur and count + w > max_words:
            pieces.append("\n".join(header + cur))
            cur, count = [], hw
        cur.append(r)
        count += w
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


def pack_blocks(blocks: list[tuple[str, str]], max_words: int) -> list[tuple[str, str]]:
    """Greedily pack atomic units into ``(kind, text)`` chunks of at most ``max_words`` words."""
    units: list[tuple[str, str]] = []
    for kind, text in blocks:
        if kind == "code":
            units.extend(("code", p) for p in split_code(text, max_words))
        elif kind == "table":
            units.extend(("table", p) for p in split_table(text, max_words))
        else:
            units.extend(("prose", p) for p in split_prose(text, max_words))
    chunks: list[tuple[str, str]] = []
    cur: list[tuple[str, str]] = []
    count = 0
    for kind, text in units:
        w = _words(text)
        if cur and count + w > max_words:
            chunks.append(_merge(cur))
            cur, count = [], 0
        cur.append((kind, text))
        count += w
    if cur:
        chunks.append(_merge(cur))
    return chunks


def _merge(units: list[tuple[str, str]]) -> tuple[str, str]:
    kinds = {k for k, _ in units}
    kind = next(iter(kinds)) if len(kinds) == 1 else "prose"
    return kind, "\n\n".join(t for _, t in units)


def chunk_document(doc: str, raw: str, max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    """Chunk one document. ``doc`` is the id, e.g. ``eip-1559``."""
    if not re.fullmatch(r"eip-\d+", doc):
        raise ValueError(f"document id {doc!r} must look like 'eip-<number>'")
    meta = front_matter(raw)
    body = raw[raw.find("\n---", 3) + 4:] if raw.startswith("---") else raw
    number = int(doc.split("-", 1)[1])
    title = meta.get("title") or doc
    chunks: list[Chunk] = []
    chunks.append(_metadata_chunk(doc, number, title, meta))
    for heading, text in split_sections(clean_markdown(body)):
        if heading.strip().lower() in _DROP_SECTIONS:
            continue
        for kind, piece in pack_blocks(split_blocks(text), max_words):
            header = f"EIP-{number}: {title} / {heading}"
            chunks.append(Chunk(
                id=f"{doc}#{len(chunks):03d}", doc=doc, number=number, title=title,
                section=heading, kind=kind, text=f"{header}\n{piece}",
            ))
    return chunks


_META_FIELDS = ("status", "type", "category", "created", "requires")


def _metadata_chunk(doc: str, number: int, title: str, meta: dict[str, str]) -> Chunk:
    """Chunk ``000``: the preamble fields that are short, factual and checkable."""
    lines = [f"eip: {number}", f"title: {title}"]
    lines += [f"{k}: {meta[k]}" for k in _META_FIELDS if meta.get(k)]
    return Chunk(
        id=f"{doc}#000", doc=doc, number=number, title=title, section="Metadata", kind="prose",
        text=f"EIP-{number}: {title} / Metadata\n" + "\n".join(lines),
    )


def chunk_corpus(docs: dict[str, str], max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, raw in docs.items():
        out.extend(chunk_document(doc, raw, max_words))
    return out
