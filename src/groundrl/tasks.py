"""Seeded task generation with strict train/val/test separation by EIP.

Question types
--------------
``constant``      a named constant, a metadata field or a blanked prose sentence
``opcode``        an opcode byte or a transaction/request type byte
``multi``         two or three facts from one EIP
``comparison``    one fact from each of two EIPs
``unanswerable``  a question whose supporting chunk is absent from the context

Leakage control: every EIP is assigned to exactly one split *before* any question is written, and
every chunk placed in a task's context (gold, hard distractor or random distractor) comes from an
EIP in the same split. Comparison questions pair two EIPs of the same split.

Invariants (enforced by tests and by :func:`validate_tasks`): in an answerable task every gold
fact has at least one supporting chunk in context; in an unanswerable task no chunk in context
supports the fact.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import Chunk
from .corpus import sha256_hex
from .facts import CLOZE, CONSTANT, METADATA, OPCODE, TYPE_BYTE, Fact, supports
from .protocol import ABSTAIN_TEXT
from .tokens import claim_tokens, content_words, word_count

SPLITS = ("train", "val", "test")
QTYPES = ("constant", "opcode", "multi", "comparison", "unanswerable")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
GENERATOR_VERSION = 1


@dataclass
class Task:
    id: str
    split: str
    qtype: str
    question: str
    context_ids: list[str]
    gold_groups: list[list[str]]  # per gold fact: every context chunk that supports it
    gold_tokens: list[str]  # normalised claim tokens of the gold values
    gold_facts: list[dict]
    gold_answer: str  # the oracle answer (or the abstention sentence)
    answerable: bool
    eips: list[str]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(**d)

    @property
    def gold_chunk_ids(self) -> list[str]:
        seen: list[str] = []
        for group in self.gold_groups:
            for cid in group:
                if cid not in seen:
                    seen.append(cid)
        return seen


class ChunkIndex:
    """Chunks by id and by document, with cached claim tokens."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("duplicate chunk ids")
        self.by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            self.by_doc[c.doc].append(c)
        self._tokens: dict[str, set[str]] = {}
        self._words: dict[str, set[str]] = {}

    def tokens(self, chunk_id: str) -> set[str]:
        if chunk_id not in self._tokens:
            self._tokens[chunk_id] = claim_tokens(self.by_id[chunk_id].text)
        return self._tokens[chunk_id]

    def words(self, chunk_id: str) -> set[str]:
        if chunk_id not in self._words:
            self._words[chunk_id] = content_words(self.by_id[chunk_id].text)
        return self._words[chunk_id]

    def supports(self, chunk_id: str, fact: Fact) -> bool:
        return supports(self.by_id[chunk_id], fact, self.tokens(chunk_id), self.words(chunk_id))

    def supporting(self, fact: Fact, chunk_ids: Iterable[str]) -> list[str]:
        return [cid for cid in chunk_ids if self.supports(cid, fact)]


def assign_splits(
    features: dict[str, dict[str, int]], seed: int,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    category_weights: dict[str, float] | None = None,
) -> dict[str, str]:
    """Assign each document to exactly one split, balancing several categories at once.

    ``features[doc]`` maps a category (``opcode``, ``named``, ``cloze``, ...) to how many facts
    of that category the document holds. Documents are visited rarest-category first (a seeded
    shuffle breaks ties) and each goes to the split whose weighted load, relative to its target
    share, would be smallest after adding it (multiway least-loaded greedy). A document is never
    divided.
    """
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must have three entries summing to 1")
    weights = category_weights or {}
    cats = sorted({c for f in features.values() for c in f} | {"docs"})
    totals = {c: sum(f.get(c, 0) for f in features.values()) + (len(features) if c == "docs" else 0)
              for c in cats}
    rng = random.Random(f"{seed}|split")
    docs = sorted(features)
    rng.shuffle(docs)
    order = [c for c in sorted(cats, key=lambda c: totals[c]) if c != "docs"]
    docs.sort(key=lambda d: tuple(-features[d].get(c, 0) for c in order))
    load = {s: dict.fromkeys(cats, 0.0) for s in SPLITS}
    out: dict[str, str] = {}
    for doc in docs:
        vec = {**dict.fromkeys(cats, 0.0), **features[doc], "docs": 1.0}
        best, best_cost = SPLITS[0], float("inf")
        for i, s in enumerate(SPLITS):
            cost = 0.0  # weighted relative load (vs. target share) after adding the document
            for c in cats:
                if totals[c] <= 0 or vec[c] <= 0:
                    continue
                cost += weights.get(c, 1.0) * (load[s][c] + vec[c]) / (ratios[i] * totals[c])
            if cost < best_cost - 1e-12:
                best, best_cost = s, cost
        out[doc] = best
        for c in cats:
            load[best][c] += vec[c]
    return out


# ---------------------------------------------------------------------------------------------
# question and answer wording


def _label(f: Fact) -> str:
    return f.name if f.name not in ("created", "requires") else f"the {f.name} field"


def _eip(f: Fact) -> str:
    return f"EIP-{f.doc.split('-', 1)[1]}"


def _single_question(f: Fact, rng: random.Random) -> str:
    n = _eip(f)
    if f.kind == CLOZE:
        return (f'In {n} (section "{f.section}"), what value fills the blank in this sentence? '
                f'"{f.blanked}"')
    if f.kind == METADATA and f.name == "created":
        return rng.choice([f"On what date was {n} created?", f"What is the creation date of {n}?"])
    if f.kind == METADATA:
        return rng.choice([f"Which EIPs does {n} list in its requires field?",
                           f"What are the prerequisites that {n} requires?"])
    if f.kind == OPCODE:
        return rng.choice([f"Which opcode byte does {n} assign to {f.name}?",
                           f"What is the opcode value of {f.name} in {n}?"])
    if f.kind == TYPE_BYTE:
        return rng.choice([f"What type byte does {f.name} have in {n}?",
                           f"In {n}, what is the value of the type byte {f.name}?"])
    return rng.choice([f"In {n}, what is the value of {f.name}?",
                       f"What value does {n} assign to {f.name}?",
                       f"According to {n}, what is {f.name} set to?"])


def _fact_sentence(f: Fact) -> str:
    """The oracle sentence for one fact, without its citation."""
    n = _eip(f)
    if f.kind == CLOZE:
        return f.sentence
    if f.kind == METADATA and f.name == "created":
        return f"{n} was created on {f.value}."
    if f.kind == METADATA:
        reqs = ", ".join(f"EIP-{x.strip()}" for x in f.value.split(","))
        return f"{n} requires {reqs}."
    if f.kind == OPCODE:
        return f"In {n}, {f.name} is opcode {f.value}."
    return f"In {n}, {f.name} is {f.value}."


def _cite(sentence: str, chunk_id: str) -> str:
    """Put the citation before the final full stop, the form a model is taught to emit."""
    body = sentence.rstrip()
    if body and body[-1] in ".!?":
        return f"{body[:-1]} [{chunk_id}]{body[-1]}"
    return f"{body} [{chunk_id}]."


def oracle_answer(facts: Sequence[Fact], citations: Sequence[str]) -> str:
    return " ".join(_cite(_fact_sentence(f), cid) for f, cid in zip(facts, citations, strict=True))


def _multi_question(facts: Sequence[Fact], rng: random.Random) -> str:
    n = _eip(facts[0])
    if all(f.kind == CLOZE for f in facts):
        blanks = " ".join(f'({chr(97 + i)}) "{f.blanked}"' for i, f in enumerate(facts))
        return f"In {n}, what values fill the blanks in these sentences? {blanks}"
    labels = [_label(f) for f in facts]
    if len(labels) == 2:
        joined = f"{labels[0]} and {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    return rng.choice([f"In {n}, what are the values of {joined}?",
                       f"Give the values that {n} specifies for {joined}."])


def _comparison_question(a: Fact, b: Fact, rng: random.Random) -> str:
    if a.kind == METADATA and a.name == "created" and b.name == "created":
        return f"Compare the creation dates of {_eip(a)} and {_eip(b)}. What are both dates?"
    return rng.choice([
        f"Compare {a.name} in {_eip(a)} with {b.name} in {_eip(b)}. What are the two values?",
        f"What value does {_eip(a)} give for {a.name}, and what value does {_eip(b)} give "
        f"for {b.name}?",
    ])


# ---------------------------------------------------------------------------------------------
# context construction


FAMILY_WEIGHTS = {"opcode": 3.0, "named": 2.0, "cloze": 1.0, "metadata": 0.5, "docs": 1.0}


def _family(f: Fact) -> str:
    if f.kind in (OPCODE, TYPE_BYTE):
        return "opcode"
    return {CONSTANT: "named", METADATA: "metadata", CLOZE: "cloze"}[f.kind]


def _qtype(f: Fact) -> str:
    return "opcode" if f.kind in (OPCODE, TYPE_BYTE) else "constant"


def build_context(
    index: ChunkIndex,
    rng: random.Random,
    facts: Sequence[Fact],
    gold_ids: Sequence[str],
    split_docs: Sequence[str],
    k: int,
    answerable: bool,
    hard_negatives: bool,
    max_words: int = 700,
) -> list[str] | None:
    """Pick ``k`` context chunk ids, or ``None`` if the invariants cannot be met.

    Distractors never support any of ``facts``. Hard negatives come from the same EIP as the
    facts; the rest come from other EIPs of the same split.
    """
    banned_names = {f.name.lower() for f in facts if f.name}
    target_docs = {f.doc for f in facts}

    def usable(c: Chunk) -> bool:
        if c.id in gold_ids:
            return False
        if banned_names & index.tokens(c.id):
            return False
        return not any(index.supports(c.id, f) for f in facts)

    same_doc = [c for d in sorted(target_docs) for c in index.by_doc[d] if usable(c)]
    other = [c for d in split_docs if d not in target_docs for c in index.by_doc[d]]
    chosen: list[str] = list(gold_ids)
    words = sum(word_count(index.by_id[c].text) for c in chosen)
    n_hard = 0
    if hard_negatives and same_doc:
        n_hard = min(len(same_doc), rng.choice([1, 2]), max(0, k - len(chosen)))
    pool_hard = rng.sample(same_doc, n_hard) if n_hard else []
    for c in pool_hard:
        chosen.append(c.id)
        words += word_count(c.text)
    attempts = 0
    while len(chosen) < k and other and attempts < 200:
        attempts += 1
        c = rng.choice(other)
        w = word_count(c.text)
        if c.id in chosen or words + w > max_words or not usable(c):
            continue
        chosen.append(c.id)
        words += w
    if len(chosen) < k:
        return None
    rng.shuffle(chosen)
    return chosen


def _make_task(
    index: ChunkIndex, rng: random.Random, split: str, facts: Sequence[Fact], qtype: str,
    question: str, answerable: bool, split_docs: Sequence[str], k: int, hard: bool, serial: int,
) -> Task | None:
    gold_ids = [f.chunk_id for f in facts] if answerable else []
    gold_ids = list(dict.fromkeys(gold_ids))
    ctx = build_context(index, rng, facts, gold_ids, split_docs, k, answerable, hard)
    if ctx is None:
        return None
    groups = [index.supporting(f, ctx) for f in facts]
    if answerable and not all(groups):
        return None
    if not answerable and any(groups):
        return None
    tokens = sorted({t for f in facts for t in f.tokens})
    if answerable:
        answer = oracle_answer(facts, [f.chunk_id for f in facts])
    else:
        answer = ABSTAIN_TEXT
    return Task(
        id=f"{split}-{serial:05d}", split=split, qtype=qtype, question=question,
        context_ids=ctx, gold_groups=groups if answerable else [[] for _ in facts],
        gold_tokens=tokens, gold_facts=[_fact_summary(f) for f in facts], gold_answer=answer,
        answerable=answerable, eips=sorted({f.doc for f in facts}),
        meta={"hard_negatives": hard, "k": k},
    )


def _fact_summary(f: Fact) -> dict:
    return {"id": f.id, "kind": f.kind, "name": f.name, "value": f.value, "doc": f.doc,
            "chunk_id": f.chunk_id}


# ---------------------------------------------------------------------------------------------
# generation


@dataclass
class GenConfig:
    seed: int = 20260
    k: int = 5
    ratios: tuple[float, float, float] = DEFAULT_RATIOS
    cloze_cap_per_doc: int = 26
    unanswerable_fraction: float = 0.16
    multi_per_doc_cap: int = 8
    comparison_per_named_fact: float = 0.9


def generate_tasks(
    chunks: Sequence[Chunk], facts: Sequence[Fact], cfg: GenConfig | None = None,
) -> dict[str, list[Task]]:
    """Generate all splits. Deterministic for a fixed ``cfg.seed`` and input."""
    cfg = cfg or GenConfig()
    index = ChunkIndex(chunks)
    capped = _cap_cloze(facts, cfg)
    features: dict[str, dict[str, int]] = {d: defaultdict(int) for d in index.by_doc}
    for f in capped:
        features[f.doc][_family(f)] += 1
    doc_split = assign_splits(features, cfg.seed, cfg.ratios, FAMILY_WEIGHTS)
    out: dict[str, list[Task]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        docs = sorted(d for d, s in doc_split.items() if s == split)
        split_facts = [f for f in capped if doc_split[f.doc] == split]
        out[split] = _generate_split(index, split, docs, split_facts, cfg)
    return out


def _cap_cloze(facts: Sequence[Fact], cfg: GenConfig) -> list[Fact]:
    rng = random.Random(f"{cfg.seed}|cap")
    by_doc: dict[str, list[Fact]] = defaultdict(list)
    keep: list[Fact] = []
    for f in facts:
        (by_doc[f.doc] if f.kind == CLOZE else keep).append(f)
    for doc in sorted(by_doc):
        pool = by_doc[doc]
        keep.extend(pool if len(pool) <= cfg.cloze_cap_per_doc
                    else rng.sample(pool, cfg.cloze_cap_per_doc))
    return sorted(keep, key=lambda f: (int(f.doc.split("-")[1]), f.chunk_id, f.id))


def _generate_split(index: ChunkIndex, split: str, docs: list[str], facts: list[Fact],
                    cfg: GenConfig) -> list[Task]:
    rng = random.Random(f"{cfg.seed}|{split}|plan")
    shuffled = list(facts)
    rng.shuffle(shuffled)
    n_un = int(round(len(shuffled) * cfg.unanswerable_fraction))
    unans_pool, ans_pool = shuffled[:n_un], shuffled[n_un:]
    tasks: list[Task] = []
    serial = 0

    def add(task: Task | None) -> None:
        nonlocal serial
        if task is not None:
            task.id = f"{split}-{serial:05d}"
            tasks.append(task)
            serial += 1

    qrng = random.Random(f"{cfg.seed}|{split}|singles")
    for f in sorted(ans_pool, key=lambda x: x.id):
        add(_make_task(index, qrng, split, [f], _qtype(f), _single_question(f, qrng), True,
                       docs, cfg.k, hard=qrng.random() < 0.6, serial=0))
    for i, f in enumerate(sorted(unans_pool, key=lambda x: x.id)):
        add(_make_task(index, qrng, split, [f], "unanswerable", _single_question(f, qrng), False,
                       docs, cfg.k, hard=i % 2 == 0, serial=0))
    mrng = random.Random(f"{cfg.seed}|{split}|multi")
    for group in _multi_groups(ans_pool, mrng, cfg.multi_per_doc_cap):
        add(_make_task(index, mrng, split, group, "multi", _multi_question(group, mrng), True,
                       docs, max(cfg.k, len(group) + 3), hard=mrng.random() < 0.6, serial=0))
    crng = random.Random(f"{cfg.seed}|{split}|comparison")
    for a, b in _comparison_pairs(ans_pool, crng, cfg.comparison_per_named_fact):
        add(_make_task(index, crng, split, [a, b], "comparison",
                       _comparison_question(a, b, crng), True, docs, cfg.k, hard=False, serial=0))
    return tasks


def _multi_groups(pool: Sequence[Fact], rng: random.Random, cap: int) -> list[list[Fact]]:
    """Groups of 2-3 facts from one EIP; named/metadata facts and cloze facts are not mixed."""
    groups: list[list[Fact]] = []
    for family in ("named", "cloze"):
        by_doc: dict[str, list[Fact]] = defaultdict(list)
        for f in pool:
            if (f.kind == CLOZE) == (family == "cloze"):
                by_doc[f.doc].append(f)
        for doc in sorted(by_doc):
            facts = sorted(by_doc[doc], key=lambda x: x.id)
            if family == "named":  # a doc's metadata facts are one "created", one "requires"
                facts = [f for f in facts if f.kind != METADATA or f.name == "requires"]
            rng.shuffle(facts)
            made = 0
            while len(facts) >= 2 and made < cap:
                size = 3 if len(facts) >= 3 and rng.random() < 0.4 else 2
                grp, facts = facts[:size], facts[size:]
                if len({f.name for f in grp}) == len(grp) or family == "cloze":
                    groups.append(sorted(grp, key=lambda x: x.chunk_id))
                    made += 1
    return groups


def _comparison_pairs(pool: Sequence[Fact], rng: random.Random,
                      per_fact: float) -> list[tuple[Fact, Fact]]:
    named = [f for f in pool if f.kind in (CONSTANT, OPCODE, TYPE_BYTE)]
    created = [f for f in pool if f.kind == METADATA and f.name == "created"]
    pairs: list[tuple[Fact, Fact]] = []
    for family, target in ((named, int(len(named) * per_fact)),
                           (created, int(len(created) * per_fact * 0.5))):
        seen: set[tuple[str, str]] = set()
        tries = 0
        while len(seen) < target and tries < target * 40 and len(family) >= 2:
            tries += 1
            a, b = rng.sample(family, 2)
            if a.doc == b.doc or a.name.lower() == b.name.lower() and a.value == b.value:
                continue
            key = tuple(sorted((a.id, b.id)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------------------------
# validation and persistence


def validate_tasks(tasks_by_split: dict[str, list[Task]], index: ChunkIndex,
                   facts_by_id: dict[str, Fact]) -> list[str]:
    """Return a list of violated invariants (empty means the task set is sound)."""
    problems: list[str] = []
    docs_by_split: dict[str, set[str]] = {s: set() for s in tasks_by_split}
    for split, tasks in tasks_by_split.items():
        for t in tasks:
            docs_by_split[split].update(c.split("#")[0] for c in t.context_ids)
            docs_by_split[split].update(t.eips)
            if t.split != split:
                problems.append(f"{t.id}: split field {t.split!r} != {split!r}")
            if len(set(t.context_ids)) != len(t.context_ids):
                problems.append(f"{t.id}: duplicate context ids")
            for gold in t.gold_facts:
                fact = facts_by_id[gold["id"]]
                groups = index.supporting(fact, t.context_ids)
                if t.answerable and not groups:
                    problems.append(f"{t.id}: no supporting chunk for {fact.id}")
                if not t.answerable and groups:
                    problems.append(f"{t.id}: unanswerable but {groups} support {fact.id}")
    splits = list(docs_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            both = docs_by_split[a] & docs_by_split[b]
            if both:
                problems.append(f"EIPs in both {a} and {b}: {sorted(both)[:5]}")
    return problems


def write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")
    return sha256_hex(body.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(
    tasks_by_split: dict[str, list[Task]], cfg: GenConfig, hashes: dict[str, str],
) -> dict:
    counts = {s: {q: 0 for q in QTYPES} for s in tasks_by_split}
    eips: dict[str, set[str]] = {}
    for s, tasks in tasks_by_split.items():
        eips[s] = set()
        for t in tasks:
            counts[s][t.qtype] += 1
            eips[s].update(t.eips)
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": cfg.seed,
        "k": cfg.k,
        "ratios": list(cfg.ratios),
        "tasks": {s: len(v) for s, v in tasks_by_split.items()},
        "total": sum(len(v) for v in tasks_by_split.values()),
        "by_type": counts,
        "answerable": {s: sum(t.answerable for t in v) for s, v in tasks_by_split.items()},
        "eips_with_questions": {s: len(e) for s, e in eips.items()},
        "sha256": hashes,
    }
