from __future__ import annotations

import json

import pytest

from groundrl.audit import POLICY_BY_NAME, run_policy
from groundrl.evaluate import (
    BASELINE_POLICIES,
    evaluate_completions,
    evaluate_policy,
    run_baselines,
)
from groundrl.protocol import ABSTAIN_TEXT
from groundrl.stats import ratio_ci
from tests.test_rewards import GOLD, comparison, ex, multi, unanswerable


def examples():
    return [ex(), multi(), comparison(), unanswerable()]


def test_oracle_completions_are_perfect():
    exs = examples()
    out = evaluate_completions(exs, [e.task.gold_answer for e in exs], n_boot=50)
    m = out["metrics"]
    assert m["grounded_correct_rate"]["mean"] == 1.0 and m["correct_rate"]["mean"] == 1.0
    assert m["verified_rate"]["mean"] == 1.0
    assert m["abstention_precision"]["value"] == 1.0 and m["abstention_recall"]["value"] == 1.0
    assert m["hallucinated_citation_rate"]["mean"] == 0.0
    assert m["mean_reward"]["v2"]["mean"] == pytest.approx(1.0, abs=0.01)
    assert out["n"] == 4 and out["n_answerable"] == 3 and out["n_unanswerable"] == 1


def test_always_abstaining_has_full_recall_and_precision_equal_to_the_unanswerable_share():
    exs = examples()
    out = evaluate_completions(exs, [ABSTAIN_TEXT] * 4, n_boot=50)
    m = out["metrics"]
    assert m["abstention_recall"]["value"] == 1.0
    assert m["abstention_precision"]["value"] == 0.25
    assert m["grounded_correct_rate"]["mean"] == 0.0 and m["abstain_rate"]["mean"] == 1.0


def test_never_abstaining_leaves_precision_undefined():
    exs = examples()
    out = evaluate_completions(exs, [GOLD] * 4, n_boot=50)
    assert out["metrics"]["abstention_precision"]["value"] is None
    assert out["metrics"]["abstention_recall"]["value"] == 0.0


def test_empty_and_malformed_completions_are_counted():
    exs = examples()
    comps = ["", "FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#3].",
             "x ​ [eip-4844#003]", "ELASTICITY_MULTIPLIER is 2 [eip-9#999]."]
    m = evaluate_completions(exs, comps, n_boot=50)["metrics"]
    assert m["empty_rate"]["mean"] == 0.25
    assert m["format_drift_rate"]["mean"] == 0.25
    assert m["suspicious_character_rate"]["mean"] == 0.25
    assert m["hallucinated_citation_rate"]["mean"] == 0.25
    assert m["invalid_citation_share"]["value"] == pytest.approx(1 / 2)


def test_correct_but_wrongly_cited_is_not_grounded_correct():
    e = ex()
    right_chunk = "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]."
    wrong_cite = "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-1559#011]."
    extra_claim = "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 (or 786432) [eip-4844#003]."
    out = evaluate_completions([e] * 3, [right_chunk, wrong_cite, extra_claim], n_boot=20)
    assert out["metrics"]["correct_rate"]["mean"] == pytest.approx(2 / 3)
    assert out["metrics"]["grounded_correct_rate"]["mean"] == pytest.approx(1 / 3)
    assert out["metrics"]["verified_rate"]["mean"] == pytest.approx(2 / 3)


def test_confidence_intervals_bracket_the_estimate():
    exs = examples() * 10
    comps = [e.task.gold_answer if i % 2 else "" for i, e in enumerate(exs)]
    m = evaluate_completions(exs, comps, n_boot=200)["metrics"]
    for key in ("grounded_correct_rate", "verified_rate", "empty_rate"):
        d = m[key]
        assert d["lo"] <= d["mean"] <= d["hi"]


def test_length_statistics_and_optional_token_counts():
    exs = examples()
    out = evaluate_completions(exs, ["one two", "a b c d", "x", ""], n_boot=20,
                               token_counts=[3, 6, 2, 0])
    words = out["metrics"]["length_words"]
    assert words["mean"] == 1.75 and words["max"] == 4
    assert out["metrics"]["length_tokens"]["max"] == 6


def test_per_question_type_breakdown():
    exs = [ex(), multi(), comparison(),
           ex(answerable=False, qtype="unanswerable", gold_groups=[[]], gold_answer=ABSTAIN_TEXT,
              context_ids=["eip-1559#011", "eip-2930#002"])]
    out = evaluate_completions(exs, [e.task.gold_answer for e in exs], n_boot=20)
    assert set(out["by_qtype"]) == {"constant", "multi", "comparison", "unanswerable"}
    assert out["by_qtype"]["unanswerable"]["grounded_correct_rate"] is None
    assert out["by_qtype"]["constant"]["grounded_correct_rate"] == 1.0


def test_chat_style_completions_and_think_blocks_are_normalised():
    out = evaluate_completions(
        [ex()], [[{"role": "assistant", "content": f"<think>x</think>{GOLD}"}]], n_boot=10)
    assert out["metrics"]["grounded_correct_rate"]["mean"] == 1.0


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="one completion per example"):
        evaluate_completions(examples(), ["a"])


def test_ratio_ci_edge_cases():
    assert ratio_ci([], [])["value"] is None
    assert ratio_ci([0, 0], [0, 0])["value"] is None
    d = ratio_ci([1, 0, 1, 1], [1, 1, 1, 1], n_boot=100)
    assert d["value"] == 0.75 and d["lo"] <= 0.75 <= d["hi"]


def test_evaluate_policy_uses_the_same_pipeline():
    exs = examples()
    out = evaluate_policy("oracle", exs, n_boot=20)
    assert out["policy"] == "oracle" and out["metrics"]["correct_rate"]["mean"] == 1.0
    spec = POLICY_BY_NAME["empty"]
    assert run_policy(spec, exs[0]) == ""
    assert evaluate_policy("empty", exs, n_boot=20)["metrics"]["empty_rate"]["mean"] == 1.0


def test_run_baselines_writes_a_labelled_file(tmp_path):
    result = run_baselines(__import__("pathlib").Path("data"), "val", tmp_path, n_boot=20, limit=15)
    assert set(result["policies"]) == set(BASELINE_POLICIES)
    on_disk = json.loads((tmp_path / "baseline_val.json").read_text())
    assert on_disk["kind"] == "baseline_eval" and on_disk["mock"] is False
    assert "no model" in on_disk["note"]
    assert on_disk["policies"]["oracle"]["metrics"]["correct_rate"]["mean"] == 1.0


def test_committed_baselines_match_a_fresh_run_for_the_deterministic_rows(dataset, root):
    from groundrl.example import make_examples

    committed = json.loads((root / "results" / "baseline_test.json").read_text())
    exs = make_examples(dataset.tasks["test"], dataset.index)
    for name in ("oracle", "refuse_all"):
        fresh = evaluate_policy(name, exs, n_boot=20)
        for key in ("grounded_correct_rate", "correct_rate", "verified_rate"):
            assert fresh["metrics"][key]["mean"] == pytest.approx(
                committed["policies"][name]["metrics"][key]["mean"])
    assert committed["split"] == "test" and committed["mock"] is False
    assert committed["policies"]["extractive"]["n"] == len(dataset.tasks["test"])
