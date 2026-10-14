from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import pytest

from groundrl.dataset import build_dataset, check_dataset, load_dataset
from groundrl.facts import extract_facts, supports
from groundrl.protocol import ABSTAIN_TEXT, CITATION
from groundrl.tasks import (
    QTYPES,
    SPLITS,
    ChunkIndex,
    GenConfig,
    Task,
    _comparison_pairs,
    _multi_groups,
    assign_splits,
    build_context,
    generate_tasks,
    read_jsonl,
    validate_tasks,
    write_jsonl,
)
from groundrl.tokens import claim_tokens
from tests.helpers import make_chunks, make_docs

NUMBERS = list(range(900, 960))


@pytest.fixture(scope="module")
def mini():
    chunks = make_chunks(NUMBERS)
    facts = extract_facts(chunks)
    cfg = GenConfig(seed=7, cloze_cap_per_doc=4)
    return chunks, facts, cfg, generate_tasks(chunks, facts, cfg)


# --- split assignment ---------------------------------------------------------------------


def test_assign_splits_puts_every_document_in_exactly_one_split():
    feats = {f"eip-{n}": {"named": n % 5, "cloze": n % 7 + 1} for n in range(100)}
    out = assign_splits(feats, seed=1)
    assert set(out) == set(feats) and set(out.values()) <= set(SPLITS)


def test_assign_splits_is_deterministic_and_seed_sensitive():
    feats = {f"eip-{n}": {"named": n % 5, "cloze": n % 7 + 1} for n in range(100)}
    assert assign_splits(feats, 1) == assign_splits(feats, 1)
    assert assign_splits(feats, 1) != assign_splits(feats, 2)


def test_assign_splits_balances_documents_and_categories():
    feats = {f"eip-{n}": {"named": n % 5, "cloze": n % 7 + 1, "opcode": 1 if n % 9 == 0 else 0}
             for n in range(200)}
    out = assign_splits(feats, 3)
    docs = Counter(out.values())
    assert 0.6 <= docs["train"] / 200 <= 0.8
    assert 15 <= docs["val"] <= 45 and 15 <= docs["test"] <= 45
    for cat in ("named", "cloze", "opcode"):
        total = sum(f[cat] for f in feats.values())
        for split, target in zip(SPLITS, (0.7, 0.15, 0.15), strict=True):
            share = sum(feats[d][cat] for d, s in out.items() if s == split) / total
            assert abs(share - target) < 0.1, (cat, split, share)


@pytest.mark.parametrize("ratios", [(0.5, 0.5), (0.6, 0.3, 0.3), (1, 0, 0.5)])
def test_assign_splits_rejects_bad_ratios(ratios):
    with pytest.raises(ValueError, match="ratios"):
        assign_splits({"eip-1": {"named": 1}}, 1, ratios)


# --- generation on a synthetic corpus ----------------------------------------------------------


def test_generation_is_deterministic(mini):
    chunks, facts, cfg, first = mini
    again = generate_tasks(chunks, facts, cfg)
    assert {s: [t.to_dict() for t in v] for s, v in again.items()} == {
        s: [t.to_dict() for t in v] for s, v in first.items()}


def test_different_seed_changes_tasks(mini):
    chunks, facts, cfg, first = mini
    other = generate_tasks(chunks, facts, GenConfig(seed=8, cloze_cap_per_doc=4))
    assert [t.question for t in other["test"]] != [t.question for t in first["test"]]


def test_all_question_types_appear_and_ids_are_sequential(mini):
    _, _, _, tasks = mini
    seen = Counter(t.qtype for v in tasks.values() for t in v)
    assert set(seen) == set(QTYPES)
    for split, v in tasks.items():
        assert [t.id for t in v] == [f"{split}-{i:05d}" for i in range(len(v))]


def test_synthetic_tasks_satisfy_every_invariant(mini):
    chunks, facts, _, tasks = mini
    assert validate_tasks(tasks, ChunkIndex(chunks), {f.id: f for f in facts}) == []


def test_no_eip_appears_in_two_splits_even_as_distractor(mini):
    _, _, _, tasks = mini
    docs = {s: {c.split("#")[0] for t in v for c in t.context_ids} | {e for t in v for e in t.eips}
            for s, v in tasks.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not docs[a] & docs[b], (a, b)


def test_validate_tasks_catches_cross_split_leakage(mini):
    chunks, facts, _, tasks = mini
    index = ChunkIndex(chunks)
    by_id = {f.id: f for f in facts}
    leaked = {s: [Task.from_dict(t.to_dict()) for t in v] for s, v in tasks.items()}
    donor = next(c for c in chunks if c.doc in {e for t in leaked["test"] for e in t.eips})
    leaked["train"][0].context_ids[0] = donor.id
    problems = validate_tasks(leaked, index, by_id)
    assert any("EIPs in both" in p for p in problems)


def test_validate_tasks_catches_missing_and_leaky_support(mini):
    chunks, facts, _, tasks = mini
    index = ChunkIndex(chunks)
    by_id = {f.id: f for f in facts}
    broken = {s: [Task.from_dict(t.to_dict()) for t in v] for s, v in tasks.items()}
    ans = next(t for t in broken["train"] if t.answerable)
    gold = ans.gold_chunk_ids[0]
    ans.context_ids = [c for c in ans.context_ids if c != gold] + ["eip-900#000"]
    unans = next(t for t in broken["val"] if not t.answerable)
    fact = by_id[unans.gold_facts[0]["id"]]
    unans.context_ids[0] = fact.chunk_id
    broken["val"][1].split = "test"
    broken["val"][2].context_ids.append(broken["val"][2].context_ids[0])
    problems = " | ".join(validate_tasks(broken, index, by_id))
    assert "no supporting chunk" in problems
    assert "unanswerable but" in problems
    assert "split field" in problems
    assert "duplicate context ids" in problems


def test_answerable_tasks_have_support_and_unanswerable_do_not(mini):
    chunks, facts, _, tasks = mini
    index = ChunkIndex(chunks)
    by_id = {f.id: f for f in facts}
    for v in tasks.values():
        for t in v:
            for gold, group in zip(t.gold_facts, t.gold_groups, strict=True):
                found = index.supporting(by_id[gold["id"]], t.context_ids)
                if t.answerable:
                    assert group and set(group) == set(found)
                else:
                    assert not found and group == []


def test_unanswerable_task_names_are_absent_from_every_context_chunk(mini):
    chunks, facts, _, tasks = mini
    index = ChunkIndex(chunks)
    by_id = {f.id: f for f in facts}
    for t in (t for v in tasks.values() for t in v if not t.answerable):
        assert t.gold_answer == ABSTAIN_TEXT
        for gold in t.gold_facts:
            name = by_id[gold["id"]].name.lower()
            if name:
                assert all(name not in index.tokens(c) for c in t.context_ids)


def test_oracle_answers_cite_only_supporting_chunks_in_context(mini):
    _, _, _, tasks = mini
    for t in (t for v in tasks.values() for t in v if t.answerable):
        cites = CITATION.findall(t.gold_answer)
        assert cites and set(cites) <= set(t.context_ids)
        assert set(cites) <= set(t.gold_chunk_ids)


def test_oracle_answers_contain_every_gold_token(mini):
    _, _, _, tasks = mini
    for t in (t for v in tasks.values() for t in v if t.answerable):
        toks = claim_tokens(t.gold_answer)
        from groundrl.facts import token_supported
        assert all(token_supported(g, toks) for g in t.gold_tokens), t.id


def test_context_size_and_composition(mini):
    chunks, _, cfg, tasks = mini
    docs_of = {c.id: c.doc for c in chunks}
    for t in (t for v in tasks.values() for t in v):
        assert len(t.context_ids) >= cfg.k
        assert len(set(t.context_ids)) == len(t.context_ids)
        assert len({docs_of[c] for c in t.context_ids}) >= 2


def test_comparison_tasks_span_two_documents_and_cite_both(mini):
    _, _, _, tasks = mini
    cmp = [t for v in tasks.values() for t in v if t.qtype == "comparison"]
    assert cmp
    for t in cmp:
        assert len(t.eips) == 2
        assert {c.split("#")[0] for c in CITATION.findall(t.gold_answer)} == set(t.eips)


def test_multi_groups_never_mix_families_or_duplicate_names():
    facts = extract_facts(make_chunks(range(900, 905)))
    groups = _multi_groups(facts, random.Random(1), cap=5)
    assert groups
    for g in groups:
        assert 2 <= len(g) <= 3 and len({f.doc for f in g}) == 1
        assert len({f.kind == "cloze" for f in g}) == 1


def test_comparison_pairs_use_different_documents():
    facts = extract_facts(make_chunks(range(900, 915)))
    pairs = _comparison_pairs(facts, random.Random(1), 1.0)
    assert pairs and all(a.doc != b.doc for a, b in pairs)
    assert len({tuple(sorted((a.id, b.id))) for a, b in pairs}) == len(pairs)


def test_build_context_returns_none_when_pool_is_too_small():
    chunks = make_chunks([900, 901])
    index = ChunkIndex(chunks)
    facts = [f for f in extract_facts(chunks) if f.doc == "eip-900"][:1]
    ctx = build_context(index, random.Random(1), facts, [facts[0].chunk_id], ["eip-900", "eip-901"],
                        k=50, answerable=True, hard_negatives=True)
    assert ctx is None


def test_build_context_respects_word_budget():
    chunks = make_chunks(NUMBERS[:10])
    index = ChunkIndex(chunks)
    fact = extract_facts(chunks)[0]
    ctx = build_context(index, random.Random(1), [fact], [fact.chunk_id],
                        sorted({c.doc for c in chunks}), k=5, answerable=True,
                        hard_negatives=False, max_words=400)
    assert ctx and sum(len(index.by_id[c].text.split()) for c in ctx) <= 400


def test_chunk_index_rejects_duplicates():
    chunks = make_chunks([900])
    with pytest.raises(ValueError, match="duplicate"):
        ChunkIndex(chunks + chunks[:1])


def test_task_roundtrip_and_gold_chunk_ids_are_unique_and_ordered():
    t = Task("t", "train", "multi", "q", ["a#001", "b#001"], [["a#001"], ["a#001", "b#001"]],
             ["1"], [], "ans", True, ["a"])
    assert Task.from_dict(t.to_dict()) == t
    assert t.gold_chunk_ids == ["a#001", "b#001"]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"b": 1, "a": "é"}, {"a": 2}]
    digest = write_jsonl(tmp_path / "x" / "r.jsonl", rows)
    assert read_jsonl(tmp_path / "x" / "r.jsonl") == rows
    assert digest == hashlib.sha256((tmp_path / "x" / "r.jsonl").read_bytes()).hexdigest()


def test_build_dataset_end_to_end_on_a_small_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for doc, text in make_docs(range(900, 940)).items():
        (raw / f"{doc}.md").write_text(text)
    summary = build_dataset(tmp_path, GenConfig(seed=3, cloze_cap_per_doc=3))
    assert summary["documents"] == 40 and summary["total"] > 100
    ds = load_dataset(tmp_path)
    assert check_dataset(ds) == []
    on_disk = json.loads((tmp_path / "tasks" / "SUMMARY.json").read_text())
    for split in SPLITS:
        digest = hashlib.sha256((tmp_path / "tasks" / f"{split}.jsonl").read_bytes()).hexdigest()
        assert on_disk["sha256"][split] == digest
    assert len(ds.context(ds.tasks["test"][0])) >= 5


# --- the committed dataset ------------------------------------------------------------------


def test_real_dataset_has_at_least_1500_questions(dataset):
    total = sum(len(v) for v in dataset.tasks.values())
    assert total >= 1500
    assert all(len(v) >= 200 for v in dataset.tasks.values())


def test_real_dataset_is_split_strictly_by_eip(dataset):
    eips = {s: {e for t in v for e in t.eips} for s, v in dataset.tasks.items()}
    ctx = {s: {c.split("#")[0] for t in v for c in t.context_ids} for s, v in dataset.tasks.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not eips[a] & eips[b], f"gold EIPs shared by {a}/{b}"
                assert not ctx[a] & ctx[b], f"context EIPs shared by {a}/{b}"
    assert all(ctx[s] >= eips[s] for s in SPLITS)


def test_real_dataset_passes_all_invariants(dataset):
    assert check_dataset(dataset) == []


def test_real_dataset_has_every_question_type_in_every_split(dataset):
    for split, v in dataset.tasks.items():
        seen = Counter(t.qtype for t in v)
        assert set(seen) == set(QTYPES), split
        assert seen["unanswerable"] >= 20 and seen["comparison"] >= 15 and seen["multi"] >= 30


def test_real_dataset_unanswerable_share_is_moderate(dataset):
    for split, v in dataset.tasks.items():
        share = sum(not t.answerable for t in v) / len(v)
        assert 0.08 <= share <= 0.20, (split, share)


def test_real_dataset_has_no_duplicate_question_text_across_splits(dataset):
    seen: dict[str, str] = {}
    for split, v in dataset.tasks.items():
        for t in v:
            prev = seen.setdefault(t.question, split)
            assert prev == split, f"question shared by {prev} and {split}: {t.question[:60]}"


def test_real_dataset_task_ids_are_unique_and_prefixed_by_split(dataset):
    ids = [t.id for v in dataset.tasks.values() for t in v]
    assert len(ids) == len(set(ids))
    for split, v in dataset.tasks.items():
        assert all(t.id.startswith(split + "-") and t.split == split for t in v)


def test_real_dataset_questions_do_not_leak_the_answer(dataset):
    for v in dataset.tasks.values():
        for t in v:
            plain_q = "fills the blank" not in t.question
            if t.qtype in ("constant", "opcode", "comparison") and plain_q:
                q_tokens = claim_tokens(t.question)
                if len(t.gold_tokens) == 1 and t.gold_tokens[0] not in ("0", "1", "2"):
                    assert t.gold_tokens[0] not in q_tokens or t.gold_tokens[0] in {
                        str(int(e.split("-")[1])) for e in t.eips}, t.id


def test_committed_task_files_match_summary_hashes(data_dir):
    summary = json.loads((data_dir / "tasks" / "SUMMARY.json").read_text())
    for split in SPLITS:
        digest = hashlib.sha256((data_dir / "tasks" / f"{split}.jsonl").read_bytes()).hexdigest()
        assert summary["sha256"][split] == digest, f"{split}.jsonl is stale; rebuild tasks"
    assert summary["total"] == sum(summary["tasks"].values())


def test_regenerating_from_the_committed_corpus_reproduces_the_committed_tasks(dataset):
    chunks = dataset.index.chunks
    facts = extract_facts(chunks)
    assert [f.to_dict() for f in facts] == [f.to_dict() for f in dataset.facts]
    again = generate_tasks(chunks, facts, GenConfig())
    for split in SPLITS:
        assert [t.to_dict() for t in again[split]] == [t.to_dict() for t in dataset.tasks[split]]


def test_every_real_gold_fact_is_supported_in_context_per_task(dataset):
    by_id = {f.id: f for f in dataset.facts}
    sample = random.Random(0).sample([t for v in dataset.tasks.values() for t in v], 300)
    for t in sample:
        for gold in t.gold_facts:
            fact = by_id[gold["id"]]
            hit = [c for c in t.context_ids
                   if supports(dataset.index.by_id[c], fact, dataset.index.tokens(c))]
            assert bool(hit) == t.answerable
