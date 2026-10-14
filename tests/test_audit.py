from __future__ import annotations

import json
import runpy
import sys

import pytest

from groundrl.audit import (
    EXPLOIT_THRESHOLD,
    HACK_NAMES,
    POLICIES,
    POLICY_BY_NAME,
    audit,
    best_chunk,
    best_unit,
    claim_sentence,
    gold_chunk,
    render_markdown,
    reward_columns,
    run_policy,
)
from groundrl.protocol import ABSTAIN_TEXT, CITATION
from groundrl.rewards import reward_v0
from groundrl.tokens import claim_tokens, suspicious_chars
from tests.test_rewards import CTX, GOLD, comparison, ex, multi, unanswerable

REQUIRED_HACKS = {
    "copy_chunk", "cite_everything", "stuff_numbers", "refuse_all", "parrot_question", "empty",
    "citation_spam", "wrong_chunk", "one_wrong_constant", "paraphrase_no_tokens",
}


def run(name, example, seed=0):
    return run_policy(POLICY_BY_NAME[name], example, seed)


def cites(text):
    return CITATION.findall(text)


def test_all_required_adversarial_policies_exist_and_names_are_unique():
    names = [p.name for p in POLICIES]
    assert len(set(names)) == len(names)
    assert REQUIRED_HACKS <= set(HACK_NAMES)
    assert {"oracle", "extractive", "eip_id_shortcut"} <= set(names)
    assert {p.category for p in POLICIES} == {"reference", "hack", "extra"}


def test_every_policy_is_deterministic_and_returns_text():
    for spec in POLICIES:
        for e in (ex(), multi(), comparison(), unanswerable()):
            assert run_policy(spec, e, 3) == run_policy(spec, e, 3)
            assert isinstance(run_policy(spec, e, 3), str)


def test_oracle_returns_the_gold_answer():
    assert run("oracle", ex()) == GOLD
    assert run("oracle", unanswerable()) == ABSTAIN_TEXT


def test_extractive_cites_one_chunk_and_never_abstains():
    out = run("extractive", ex())
    assert cites(out) == ["eip-4844#003"] and "4096" in out
    assert run("extractive", unanswerable()) != ABSTAIN_TEXT


def test_eip_id_shortcut_abstains_only_when_the_named_eip_is_absent():
    assert run("eip_id_shortcut", ex()) == run("extractive", ex())
    absent = ex(question="In EIP-7002, what is the value of X?")
    assert run("eip_id_shortcut", absent) == ABSTAIN_TEXT


def test_copy_chunk_copies_every_line_of_the_gold_chunk_with_its_citation():
    out = run("copy_chunk", ex())
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) >= 4 and all(cites(ln) == ["eip-4844#003"] for ln in lines)
    assert "FIELD_ELEMENTS_PER_BLOB" in out and "786432" in out
    assert len(out.split()) > 1.5 * len(GOLD.split())


def test_cite_everything_keeps_the_answer_but_cites_all_context_chunks_per_sentence():
    out = run("cite_everything", multi())
    for sentence in out.split(". "):
        assert set(cites(sentence)) == set(CTX) or not cites(sentence)
    assert "4096" in out and "2**17" in out and len(set(cites(out))) == len(CTX)


def test_stuff_numbers_lists_every_checkable_token_of_the_gold_chunk_in_one_sentence():
    out = run("stuff_numbers", ex())
    assert cites(out) == ["eip-4844#003"] and out.count("[") == 1
    for token in ("4096", "2**17", "786432", "0x03"):
        assert token in out


def test_refuse_all_always_abstains():
    assert run("refuse_all", ex()) == ABSTAIN_TEXT == run("refuse_all", unanswerable())


def test_parrot_question_restates_the_question_without_a_question_mark():
    out = run("parrot_question", ex())
    assert "what is the value of FIELD_ELEMENTS_PER_BLOB" in out and "?" not in out
    assert len(cites(out)) == 1


def test_empty_policy_is_empty():
    assert run("empty", ex()) == ""


def test_citation_spam_repeats_one_sentence_twelve_times_with_triple_citations():
    out = run("citation_spam", ex())
    sentences = [s for s in out.split(". ") if s]
    assert len(sentences) == 12
    assert out.count("[eip-4844#003]") == 36


def test_wrong_chunk_cites_a_chunk_that_is_not_gold():
    for seed in range(5):
        out = run("wrong_chunk", ex(), seed)
        assert cites(out)[0] != "eip-4844#003" and cites(out)[0] in CTX


def test_one_wrong_constant_swaps_the_gold_value_for_another_value_from_the_same_chunk():
    for seed in range(8):
        out = run("one_wrong_constant", ex(), seed)
        assert "4096" not in out and cites(out) == ["eip-4844#003"]
        new = (claim_tokens(out) - claim_tokens(GOLD)) - {"4096"}
        assert len(new) == 1 and new <= claim_tokens(CTX["eip-4844#003"])
        assert reward_v0(out, ex())[0] == 1.0  # the verifier cannot tell


def test_one_wrong_constant_in_a_multi_fact_answer_keeps_the_other_fact_correct():
    out = run("one_wrong_constant", multi())
    kept = [t for t in ("4096", "2**17") if t in out]
    assert len(kept) == 1


def test_one_wrong_constant_on_unanswerable_falls_back_to_a_guess():
    assert run("one_wrong_constant", unanswerable()) != ABSTAIN_TEXT


def test_paraphrase_has_no_checkable_token_and_cites_the_gold_chunk():
    out = run("paraphrase_no_tokens", ex())
    assert claim_tokens(out) == set() and cites(out) == ["eip-4844#003"]
    assert reward_v0(out, ex())[0] == 1.0  # the verifier's known blind spot


def test_hedge_right_and_wrong_contains_the_correct_sentence_and_a_contradiction():
    out = run("hedge_right_and_wrong", ex())
    assert out.startswith(GOLD) and out.count("FIELD_ELEMENTS_PER_BLOB") == 2
    assert "4096" in out and reward_v0(out, ex())[0] == 1.0


def test_hedge_inline_adds_an_alternative_value_in_the_same_sentence():
    out = run("hedge_inline", ex())
    assert "(or " in out and "4096" in out and len(cites(out)) == 1


def test_spoofed_citation_points_at_a_chunk_index_that_does_not_exist():
    out = run("spoofed_citation", multi())
    assert cites(out) and all(c.endswith("#999") for c in cites(out))


def test_hallucinated_value_is_found_nowhere_in_the_context():
    for seed in range(5):
        out = run("hallucinated_value", ex(), seed)
        invented = claim_tokens(out) - claim_tokens(GOLD)
        assert invented and not any(invented & claim_tokens(t) for t in CTX.values())
        assert cites(out) == ["eip-4844#003"]


def test_uncited_correct_is_the_gold_answer_without_citations():
    out = run("uncited_correct", ex())
    assert cites(out) == [] and "4096" in out and "[" not in out


def test_unicode_obfuscation_uses_fullwidth_digits_and_keeps_the_citation_intact():
    out = run("unicode_obfuscation", ex())
    assert "４０９６" in out and cites(out) == ["eip-4844#003"]
    assert suspicious_chars(out)


def test_helpers_pick_the_expected_chunks():
    e = ex()
    line, cid = best_unit(e)
    assert cid == "eip-4844#003" and "FIELD_ELEMENTS_PER_BLOB" in line
    assert best_chunk(e) == "eip-4844#003" and gold_chunk(e) == "eip-4844#003"
    assert gold_chunk(unanswerable()) == best_chunk(unanswerable())
    assert claim_tokens(claim_sentence(e, "eip-1559#011"))


# --- the audit runner ---


@pytest.fixture(scope="module")
def small_audit():
    return audit([ex(), multi(), comparison(), unanswerable()], n_boot=50)


def test_audit_reports_every_policy_under_every_reward_column(small_audit):
    cols = list(reward_columns())
    assert small_audit["reward_columns"] == cols
    assert {"v0", "v1", "v2", "v2-rev1", "v2-gold-free"} <= set(cols)
    assert {c for c in cols if c.startswith("v2-no-")} == {
        f"v2-no-{t}" for t in ("abstention", "correctness", "citation", "relevance", "length",
                                "redundancy", "consistency")}
    assert set(small_audit["policies"]) == {p.name for p in POLICIES}
    for entry in small_audit["policies"].values():
        assert set(entry["scores"]) == set(cols)
        for s in entry["scores"].values():
            assert 0.0 <= s["mean"] <= 1.0 and s["lo"] <= s["mean"] <= s["hi"]


def test_audit_counts_and_threshold(small_audit):
    assert small_audit["n_tasks"] == 4 and small_audit["n_answerable"] == 3
    assert small_audit["n_unanswerable"] == 1
    assert small_audit["exploit_threshold"] == EXPLOIT_THRESHOLD == 0.5


def test_audit_never_flags_reference_policies_and_flags_known_v0_exploits(small_audit):
    for col, names in small_audit["exploitable"].items():
        assert not {"oracle", "extractive", "eip_id_shortcut"} & set(names), col
    v0 = set(small_audit["exploitable"]["v0"])
    assert {"cite_everything", "stuff_numbers", "paraphrase_no_tokens", "citation_spam"} <= v0
    assert small_audit["exploitable"]["v2"] == []


def test_audit_is_serialisable_and_reproducible(small_audit):
    again = audit([ex(), multi(), comparison(), unanswerable()], n_boot=50)
    assert json.dumps(small_audit, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_audit_records_how_often_each_trick_fires(small_audit):
    assert small_audit["policies"]["oracle"]["applied_fraction"] == 0.0
    assert small_audit["policies"]["empty"]["applied_fraction"] == 1.0
    assert small_audit["policies"]["oracle"]["scores"]["v2"]["applied_mean"] is None
    assert small_audit["policies"]["empty"]["scores"]["v2"]["applied_mean"] == 0.0


def test_audit_slices_answerable_and_unanswerable(small_audit):
    refuse = small_audit["policies"]["refuse_all"]["scores"]["v2"]
    assert refuse["answerable_mean"] == 0.0 and refuse["unanswerable_mean"] == 1.0


def test_render_markdown_contains_both_tables_and_marks_exploits(small_audit):
    md = render_markdown(small_audit, "test")
    assert "| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |" in md
    assert "no consistency" in md and "gold-free" in md and "rev1" in md
    assert "**1.000**" in md and "Exploitable" in md
    assert md.count("| `oracle` |") == 1  # reference rows only in the first table


def test_audit_subset_of_policies():
    result = audit([ex()], [POLICY_BY_NAME["empty"], POLICY_BY_NAME["oracle"]], n_boot=10)
    assert list(result["policies"]) == ["empty", "oracle"]


def test_reward_audit_script_writes_json_and_markdown(tmp_path, root, monkeypatch, capsys):
    out = tmp_path / "res"
    argv = ["reward_audit.py", "--split", "val", "--limit", "12", "--boot", "20",
            "--out", str(out), "--policies", "oracle", "empty", "refuse_all"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / "scripts" / "reward_audit.py"), run_name="__main__")
    assert exc.value.code == 0
    data = json.loads((out / "audit_val.json").read_text())
    assert data["split"] == "val" and data["n_tasks"] == 12 and len(data["policies"]) == 3
    assert data["command"].endswith("--split val")
    assert (out / "audit_val.md").read_text().startswith("Reward-hacking audit on the VAL split")
    assert "oracle" in capsys.readouterr().out


# --- the committed audit ---


def test_committed_audit_matches_the_dataset_and_its_cheap_rows_reproduce(root, dataset):
    from groundrl.example import make_examples

    path = root / "results" / "audit_test.json"
    data = json.loads(path.read_text())
    summary = json.loads((root / "data" / "tasks" / "SUMMARY.json").read_text())
    assert data["kind"] == "reward_audit" and data["split"] == "test"
    assert data["task_file_sha256"] == summary["sha256"]["test"]
    assert data["n_tasks"] == len(dataset.tasks["test"])
    examples = make_examples(dataset.tasks["test"], dataset.index)
    cheap = [POLICY_BY_NAME[n] for n in ("oracle", "empty", "refuse_all", "spoofed_citation")]
    fresh = audit(examples, cheap, n_boot=50)
    for name, entry in fresh["policies"].items():
        for col in ("v0", "v1", "v2"):
            assert entry["scores"][col]["mean"] == pytest.approx(
                data["policies"][name]["scores"][col]["mean"]), (name, col)


def test_committed_audit_satisfies_the_preregistered_claims(root):
    data = json.loads((root / "results" / "audit_test.json").read_text())
    pol = data["policies"]
    assert pol["oracle"]["scores"]["v2"]["mean"] == pytest.approx(1.0, abs=0.01)
    assert pol["empty"]["scores"]["v2"]["mean"] == 0.0
    assert set(REQUIRED_HACKS) <= set(pol)
    # the point of the revisions: v0 and v1 are exploitable, final v2 is not
    assert len(data["exploitable"]["v0"]) >= 6 and len(data["exploitable"]["v1"]) >= 6
    assert data["exploitable"]["v2"] == []


def test_closed_by_names_the_terms_whose_removal_reopens_a_policy(small_audit):
    from groundrl.audit import closed_by

    assert small_audit["closed_by"] == closed_by(small_audit)
    assert "oracle" not in small_audit["closed_by"]
    synthetic = {"policies": {
        "ref": {"category": "reference", "scores": {"v2": {"mean": 1.0}}},
        "open": {"category": "hack", "scores": {"v2": {"mean": 0.9}}},
        "shut": {"category": "hack", "scores": {
            "v2": {"mean": 0.1}, "v2-no-citation": {"mean": 0.8}, "v2-no-length": {"mean": 0.2}}},
    }}
    assert closed_by(synthetic) == {"shut": ["citation"]}


def test_markdown_lists_the_closing_terms(small_audit):
    md = render_markdown(small_audit, "test")
    assert "Held below 0.50 by exactly these v2 terms" in md
