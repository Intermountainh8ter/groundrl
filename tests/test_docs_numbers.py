"""Numbers quoted in the documentation must match the committed result files."""
from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def results(root):
    return {
        "test": json.loads((root / "results" / "audit_test.json").read_text()),
        "val": json.loads((root / "results" / "audit_val.json").read_text()),
        "baseline": json.loads((root / "results" / "baseline_test.json").read_text()),
    }


def score(audit, policy, column="v2"):
    return audit["policies"][policy]["scores"][column]["mean"]


def test_methodology_quotes_the_audit_numbers_it_was_written_from(root, results):
    doc = (root / "docs" / "METHODOLOGY.md").read_text()
    t, v, b = results["test"], results["val"], results["baseline"]
    ext = b["policies"]["extractive"]["metrics"]["mean_reward"]["v2"]
    expected = [
        f"TEST split, {t['n_tasks']} tasks)",
        f"`parrot_question` ({score(t, 'parrot_question', 'v1'):.3f} on TEST, "
        f"{score(v, 'parrot_question', 'v1'):.3f} on VAL)",
        f"mean {score(t, 'hedge_right_and_wrong', 'v2-rev1'):.3f}",
        f"mean {score(t, 'hedge_inline', 'v2-rev1'):.3f}",
        f"`hedge_inline` at {score(t, 'hedge_inline'):.3f} (TEST) and "
        f"{score(v, 'hedge_inline'):.3f} (VAL)",
        f"its TEST mean is {t['policies']['hedge_inline']['scores']['v2']['applied_mean']:.3f}",
        f"`hallucinated_value` ({score(t, 'hallucinated_value', 'v1'):.2f})",
        f"`cite_everything` is back at {score(t, 'cite_everything', 'v2-no-citation'):.2f}",
        f"`stuff_numbers` ({score(t, 'stuff_numbers', 'v2-no-correctness'):.2f})",
        f"`one_wrong_constant` ({score(t, 'one_wrong_constant', 'v2-no-correctness'):.2f})",
        f"`hallucinated_value` ({score(t, 'hallucinated_value', 'v2-no-correctness'):.2f})",
        f"(it is {score(t, 'wrong_chunk', 'v2-gold-free'):.2f} gold-free)",
        f"`parrot_question` is at {score(t, 'parrot_question', 'v2-no-correctness'):.2f}",
        f"`paraphrase_no_tokens` at {score(t, 'paraphrase_no_tokens', 'v2-no-correctness'):.2f}",
        f"(0.{round(score(t, 'refuse_all') * 1000):03d}) instead of 0",
        f"exploitable by {len(t['exploitable']['v2-gold-free'])} policies on\n  TEST and "
        f"{len(v['exploitable']['v2-gold-free'])} on VAL",
        f"reaches {ext['mean']:.3f} (95% CI {ext['lo']:.3f}-{ext['hi']:.3f}) on TEST",
    ]
    missing = [e for e in expected if e not in doc]
    assert not missing, missing


def test_methodology_claims_about_which_policies_are_exploitable(root, results):
    t = results["test"]
    required = {"copy_chunk", "cite_everything", "stuff_numbers", "citation_spam", "wrong_chunk",
                "one_wrong_constant", "paraphrase_no_tokens"}
    assert required <= set(t["exploitable"]["v0"])
    assert t["exploitable"]["v2"] == [] and results["val"]["exploitable"]["v2"] == []
    assert set(t["exploitable"]["v2-rev1"]) == {"hedge_right_and_wrong", "hedge_inline"}
    assert set(t["exploitable"]["v1"]) - set(t["exploitable"]["v0"]) == {
        "hallucinated_value", "parrot_question"}
    assert {k for k, vv in t["closed_by"].items() if vv} == {
        "cite_everything", "stuff_numbers", "one_wrong_constant", "hedge_inline",
        "hallucinated_value"}
    share = t["policies"]["copy_chunk"]["scores"]["v2"]["frac_ge_0.5"]
    assert round(share * 100) == 14


def _first_table(markdown: str) -> list[str]:
    lines = markdown.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("| policy | kind"))
    out = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        out.append(line)
    return out


def test_readme_audit_table_is_the_committed_audit_table(root):
    readme = (root / "README.md").read_text()
    for line in _first_table((root / "results" / "audit_test.md").read_text()):
        assert line in readme, line


def test_readme_baseline_table_matches_the_baseline_file(root, results):
    readme = (root / "README.md").read_text()
    for name, p in results["baseline"]["policies"].items():
        m = p["metrics"]
        v2 = m["mean_reward"]["v2"]
        assert f"{v2['mean']:.3f} ({v2['lo']:.3f}-{v2['hi']:.3f})" in readme, name


def test_readme_claims_and_counts(root, results):
    readme = (root / "README.md").read_text()
    summary = json.loads((root / "data" / "tasks" / "SUMMARY.json").read_text())
    assert "Training results: NOT YET RUN" in readme
    for key in ("train", "val", "test"):
        assert f"`{summary['tasks'][key]}` {key}" in readme
    assert f"{summary['total']} seeded questions" in readme
    t = results["test"]
    split_line = (f"({t['n_tasks']} tasks: {t['n_answerable']} answerable, "
                  f"{t['n_unanswerable']} unanswerable)")
    assert split_line in readme
    assert f"{score(t, 'hedge_inline'):.3f}" in readme


def test_diagram_text_quotes_current_counts(root):
    summary = json.loads((root / "data" / "tasks" / "SUMMARY.json").read_text())
    usage = (root / "docs" / "diagrams" / "usage.mmd").read_text()
    from groundrl.audit import POLICIES

    assert f"{summary['total']} tasks" in usage and f"{len(POLICIES)} policies" in usage
    assert f"{summary['documents']} EIPs" in usage
    flowcharts = (root / "docs" / "FLOWCHARTS.md").read_text()
    assert usage.strip() in flowcharts
    assert (root / "docs" / "diagrams" / "troubleshooting.mmd").read_text().strip() in flowcharts
    assert flowcharts.count("```mermaid") == 2


def test_threats_quote_current_corpus_and_budget_numbers(root, dataset):
    from collections import Counter

    from groundrl.trainconf import load_config
    from groundrl.trainplan import build_plan

    doc = (root / "docs" / "THREATS_TO_VALIDITY.md").read_text()
    kinds = Counter(f.kind for f in dataset.facts)
    assert f"{kinds['opcode'] + kinds['type_byte']} opcode and type-byte facts" in doc
    test_opcode = sum(t.qtype == "opcode" for t in dataset.tasks["test"])
    assert f"{test_opcode} TEST\n  tasks" in doc
    expected = (("qwen3-0.6b-24gb", "0.7% at 2048"), ("qwen3-0.6b-12gb", "2.8% at\n  1536"))
    for name, tokens in expected:
        plan = build_plan(load_config(root / "train" / "configs" / f"{name}.json"), "sft",
                          repo_root=root)
        share = 100 * plan["dropped"] / plan["examples"]
        assert f"{share:.1f}%" in tokens
        assert tokens in doc
