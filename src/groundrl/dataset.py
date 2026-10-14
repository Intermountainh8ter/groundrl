"""Build and load the derived dataset: chunks, facts and per-split task files.

Layout under ``data/``::

    raw/eip-N.md            downloaded proposals (CC0), checked by MANIFEST.json
    MANIFEST.json           sha256 + provenance per raw file
    chunks.jsonl            every chunk of every document (derived, deterministic)
    facts.jsonl             every extracted fact (derived, deterministic)
    tasks/{train,val,test}.jsonl   tasks, referring to chunks by id only
    tasks/SUMMARY.json      counts, seed and sha256 of each task file
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, chunk_corpus
from .corpus import load_documents
from .facts import Fact, extract_facts
from .tasks import (
    SPLITS,
    ChunkIndex,
    GenConfig,
    Task,
    generate_tasks,
    read_jsonl,
    summarize,
    validate_tasks,
    write_jsonl,
)


@dataclass
class Dataset:
    index: ChunkIndex
    facts: list[Fact]
    tasks: dict[str, list[Task]]

    def context(self, task: Task) -> list[Chunk]:
        return [self.index.by_id[cid] for cid in task.context_ids]


def build_dataset(data_dir: Path, cfg: GenConfig | None = None) -> dict:
    """Chunk the raw corpus, extract facts, generate and validate tasks; write everything."""
    data_dir = Path(data_dir)
    cfg = cfg or GenConfig()
    docs = load_documents(data_dir / "raw")
    chunks = chunk_corpus(docs)
    facts = extract_facts(chunks)
    tasks = generate_tasks(chunks, facts, cfg)
    problems = validate_tasks(tasks, ChunkIndex(chunks), {f.id: f for f in facts})
    if problems:
        raise ValueError("generated tasks violate invariants: " + "; ".join(problems[:5]))
    write_jsonl(data_dir / "chunks.jsonl", (c.to_dict() for c in chunks))
    write_jsonl(data_dir / "facts.jsonl", (f.to_dict() for f in facts))
    hashes = {s: write_jsonl(data_dir / "tasks" / f"{s}.jsonl", (t.to_dict() for t in tasks[s]))
              for s in SPLITS}
    summary = summarize(tasks, cfg, hashes)
    summary["documents"] = len(docs)
    summary["chunks"] = len(chunks)
    summary["facts"] = len(facts)
    (data_dir / "tasks" / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_dataset(data_dir: Path) -> Dataset:
    data_dir = Path(data_dir)
    chunks = [Chunk.from_dict(d) for d in read_jsonl(data_dir / "chunks.jsonl")]
    facts = [Fact.from_dict(d) for d in read_jsonl(data_dir / "facts.jsonl")]
    tasks = {s: [Task.from_dict(d) for d in read_jsonl(data_dir / "tasks" / f"{s}.jsonl")]
             for s in SPLITS}
    return Dataset(ChunkIndex(chunks), facts, tasks)


def check_dataset(ds: Dataset) -> list[str]:
    """Re-validate a loaded dataset (used by tests and ``groundrl check-data``)."""
    return validate_tasks(ds.tasks, ds.index, {f.id: f for f in ds.facts})
