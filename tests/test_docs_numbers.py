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
    # SFT has run on gfx906; GRPO has been run and the run was degenerate
    # (null result, all 300 steps produced zero reward and zero completion
    # length). The README advertises both.
    assert "SFT on Qwen3-0.6B" in readme
    assert "| GRPO, reward v2 | DEGENERATE" in readme
    assert "GRPO run was degenerate" in readme
    assert "results/groundrl-grpo/summary_grpo.json" in readme
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


def _pct(x):
    return f"{x * 100:.1f}"


def _interval(metric, key="mean"):
    return f"{_pct(metric[key])}% ({_pct(metric['lo'])}-{_pct(metric['hi'])})"


def _training_files(root):
    sft_dir = root / "results" / "groundrl-sft"
    return (json.loads((sft_dir / "eval_baseline.json").read_text()),
            json.loads((sft_dir / "eval_sft.json").read_text()),
            json.loads((sft_dir / "summary_sft.json").read_text()),
            json.loads((root / "results" / "groundrl-grpo" / "summary_grpo.json").read_text()))


def test_readme_training_table_matches_the_eval_files(root):
    readme = " ".join((root / "README.md").read_text().split())
    base, sft, train, _ = _training_files(root)
    for data in (base, sft):
        m = data["metrics"]
        v2 = m["mean_reward"]["v2"]
        cells = [_interval(m["grounded_correct_rate"]), _interval(m["verified_rate"]),
                 _interval(m["hallucinated_citation_rate"]),
                 f"{_interval(m['abstention_precision'], 'value')} / "
                 f"{_interval(m['abstention_recall'], 'value')}",
                 f"{v2['mean']:.3f} ({v2['lo']:.3f}-{v2['hi']:.3f})"]
        for cell in cells:
            assert cell in readme, cell
    steps = max(r["step"] for r in train["log_history"] if "step" in r)
    final = train["log_history"][-1]
    assert f"{steps} optimizer steps, {train['seconds'] / 60:.1f} min" in readme
    assert f"{final['train_loss']:.4f}" in readme
    assert f"{train['train_examples']:,} training examples" in readme
    assert f"{train['dropped_by_prompt_budget']} prompts over the token budget" in readme


def test_readme_and_docs_describe_the_grpo_run_as_the_log_shows(root):
    _, _, _, grpo = _training_files(root)
    rows = [r for r in grpo["log_history"] if "completions/mean_length" in r]
    assert rows and all(r["completions/mean_length"] == 0 and r["reward"] == 0
                        and r["grad_norm"] == 0 for r in rows)
    steps = max(r["step"] for r in grpo["log_history"] if "step" in r)
    readme = " ".join((root / "README.md").read_text().split())
    assert f"across all {steps} optimizer steps" in readme
    assert "null result" in readme and "not shown to work or to fail" in readme
    threats = " ".join((root / "docs" / "THREATS_TO_VALIDITY.md").read_text().split())
    assert f"All {steps} steps reported zero reward, zero completion length" in threats


def test_docs_do_not_say_nothing_has_been_trained(root):
    for rel in ("README.md", "docs/THREATS_TO_VALIDITY.md", "docs/RUNBOOK.md",
                "docs/FLOWCHARTS.md", "docs/diagrams/usage.mmd", "docs/METHODOLOGY.md"):
        low = " ".join((root / rel).read_text().split()).lower()
        for stale in ("unrun", "nothing has been trained", "no model has been trained",
                      "never been run", "never run in this repository", "training is unrun"):
            assert stale not in low, (rel, stale)


def _template(question: str) -> str:
    """The question with quoted text, EIP numbers, identifiers, hex values and numbers masked."""
    import re

    q = re.sub(r'"[^"]*"', '"X"', question)
    q = re.sub(r"EIP-\d+", "EIP-N", q)
    q = re.sub(r"\b0x[0-9a-fA-F]+\b", "H", q)
    q = re.sub(r"\b[A-Z][A-Z0-9_]{2,}\b", "ID", q)
    return re.sub(r"\d+", "N", q)


def test_threats_quote_the_template_overlap_and_base_model_numbers(root):
    from collections import Counter

    def questions(split):
        path = root / "data" / "tasks" / f"{split}.jsonl"
        return [json.loads(line)["question"] for line in path.read_text().splitlines()]

    train, test = Counter(map(_template, questions("train"))), Counter(
        map(_template, questions("test")))
    assert set(test) <= set(train)
    blank = max(test.values())
    doc = " ".join((root / "docs" / "THREATS_TO_VALIDITY.md").read_text().split())
    assert (f"every one of the {sum(test.values())} TEST questions follows a template that also "
            f"occurs in TRAIN ({len(test)} templates in TEST, {len(train)} in TRAIN); {blank} of "
            f"the {sum(test.values())} use the single") in doc
    base, _, _, _ = _training_files(root)
    m = base["metrics"]
    assert f"abstained on {_pct(m['abstain_rate']['mean'])}% of TEST questions" in doc
    assert f"only {_pct(base['n_unanswerable'] / base['n'])}% are unanswerable" in doc
    assert f"drifted from the answer format on {_pct(m['format_drift_rate']['mean'])}%" in doc
