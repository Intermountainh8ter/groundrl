"""A self-contained scoring/training example: a task plus the text of its context chunks."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from .tasks import ChunkIndex, Task


@dataclass
class Example:
    task: Task
    context: dict[str, str]  # chunk id -> chunk text, in prompt order

    @property
    def context_ids(self) -> list[str]:
        return list(self.context)

    def to_json(self) -> str:
        return json.dumps({"task": self.task.to_dict(),
                           "context": [{"id": k, "text": v} for k, v in self.context.items()]},
                          sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> Example:
        d = json.loads(blob)
        return cls(Task.from_dict(d["task"]), {c["id"]: c["text"] for c in d["context"]})


def make_example(task: Task, index: ChunkIndex) -> Example:
    return Example(task, {cid: index.by_id[cid].text for cid in task.context_ids})


def make_examples(tasks: Iterable[Task], index: ChunkIndex) -> list[Example]:
    return [make_example(t, index) for t in tasks]
