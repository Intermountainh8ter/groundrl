from __future__ import annotations

import random
import string
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from groundrl.example import Example
from groundrl.protocol import ABSTAIN_TEXT
from groundrl.rewards import (
    V2_TERMS,
    V2Config,
    citation_quality,
    completion_text,
    contradictions,
    correctness,
    graded_grounding,
    length_factor,
    make_trl_reward,
    redundancy_factor,
    relevance,
    reward_v0,
    reward_v1,
    reward_v2,
    score_completion,
    strip_think,
    token_equiv,
)
from groundrl.tasks import Task
from groundrl.verifier import verify_answer

ROOT = Path(__file__).resolve().parents[1]

BLOB = ("EIP-4844: Shard Blob Transactions / Parameters\n"
        "| `FIELD_ELEMENTS_PER_BLOB` | `4096` |\n| `GAS_PER_BLOB` | `2**17` |\n"
        "| `MAX_BLOB_GAS_PER_BLOCK` | `786432` |\n| `BLOB_TX_TYPE` | `Bytes1(0x03)` |")
FEE = ("EIP-1559: Fee market change / Specification\n"
       "INITIAL_BASE_FEE = 1000000000\nELASTICITY_MULTIPLIER = 2")
ACL = "EIP-2930: Optional access lists / Parameters\nACCESS_LIST_ADDRESS_COST = 2400"
CTX = {"eip-4844#003": BLOB, "eip-1559#011": FEE, "eip-2930#002": ACL}
GOLD = "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]."


def task(**kw) -> Task:
    base = dict(
        id="t-1", split="test", qtype="constant",
        question="In EIP-4844, what is the value of FIELD_ELEMENTS_PER_BLOB?",
        context_ids=list(CTX), gold_groups=[["eip-4844#003"]], gold_tokens=["4096"],
        gold_facts=[], gold_answer=GOLD, answerable=True, eips=["eip-4844"])
    base.update(kw)
    return Task(**base)


def ex(**kw) -> Example:
    return Example(task(**kw), dict(CTX))


def unanswerable() -> Example:
    return ex(answerable=False, gold_groups=[[]], gold_answer=ABSTAIN_TEXT,
              context_ids=["eip-1559#011", "eip-2930#002"])


def multi() -> Example:
    return ex(
        qtype="multi",
        question="In EIP-4844, what are the values of FIELD_ELEMENTS_PER_BLOB and GAS_PER_BLOB?",
        gold_groups=[["eip-4844#003"], ["eip-4844#003"]], gold_tokens=["2**17", "4096"],
        gold_answer="In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]. "
                    "In EIP-4844, GAS_PER_BLOB is 2**17 [eip-4844#003].")


def comparison() -> Example:
    return ex(
        qtype="comparison", eips=["eip-1559", "eip-4844"],
        question="Compare ELASTICITY_MULTIPLIER in EIP-1559 with BLOB_TX_TYPE in EIP-4844.",
        gold_groups=[["eip-1559#011"], ["eip-4844#003"]], gold_tokens=["0x3", "2"],
        gold_answer="In EIP-1559, ELASTICITY_MULTIPLIER is 2 [eip-1559#011]. "
                    "In EIP-4844, BLOB_TX_TYPE is 0x03 [eip-4844#003].")


# --- helpers ---


def test_strip_think_removes_closed_blocks_and_truncates_open_ones():
    assert strip_think("<think>hmm</think>answer") == "answer"
    assert strip_think("a<think>x</think>b<think>y</think>c") == "abc"
    assert strip_think("answer<think>never closes") == "answer"
    assert strip_think("plain") == "plain"


def test_completion_text_handles_strings_messages_and_junk():
    assert completion_text("x") == "x"
    assert completion_text([{"role": "assistant", "content": "ans"}]) == "ans"
    assert completion_text([{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "ans"}]) == "ans"
    assert completion_text({"role": "assistant", "content": "<think>t</think>ok"}) == "ok"
    assert completion_text(None) == "None"
    assert completion_text([]) == ""


def test_token_equiv_is_symmetric():
    assert token_equiv("0x40", "64") and token_equiv("64", "0x40") and token_equiv("2**6", "64")
    assert token_equiv("64", "2**6") and token_equiv("2", "0x2") and token_equiv("0x2", "2")
    assert token_equiv("a", "a") and not token_equiv("a", "b") and not token_equiv("4", "5")
    assert not token_equiv("a", "4")


# --- v0 / v1 ---


def test_v0_is_binary():
    assert reward_v0(GOLD, ex())[0] == 1.0
    assert reward_v0("FIELD_ELEMENTS_PER_BLOB is 8192 [eip-4844#003].", ex())[0] == 0.0
    s, bd = reward_v0(GOLD, ex())
    assert bd["verified"] is True and bd["version"] == "v0"


def test_v0_scores_honest_abstention_as_zero():
    assert reward_v0(ABSTAIN_TEXT, unanswerable())[0] == 0.0


def test_v1_gives_full_credit_to_a_grounded_answer_and_zero_to_nothing():
    assert reward_v1(GOLD, ex())[0] == pytest.approx(1.0, abs=0.02)
    assert reward_v1("", ex())[0] == 0.0
    assert reward_v1("No citation here 4096.", ex())[0] == 0.0


def test_v1_gives_partial_credit_for_partially_supported_sentences():
    full = reward_v1("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003].", ex())[0]
    half = reward_v1("FIELD_ELEMENTS_PER_BLOB is 8192 [eip-4844#003].", ex())[0]
    none = reward_v1("OTHER_THING is 8192 [eip-4844#003].", ex())[0]
    assert full > half > none >= 0.0
    assert 0.0 < half < 1.0


def test_v1_averages_over_sentences():
    one_bad = reward_v1("GAS_PER_BLOB is 2**17 [eip-4844#003]. Uncited sentence here.", ex())[0]
    assert 0.4 < one_bad < 0.6


def test_v1_penalises_invalid_citations_but_keeps_some_credit():
    good = reward_v1("GAS_PER_BLOB is 2**17 [eip-4844#003].", ex())[0]
    mixed = reward_v1("GAS_PER_BLOB is 2**17 [eip-4844#003][eip-4844#999].", ex())[0]
    spoof = reward_v1("GAS_PER_BLOB is 2**17 [eip-4844#999].", ex())[0]
    assert good > mixed > spoof == 0.0


def test_v1_is_zero_for_suspicious_characters_and_empty_answers():
    res = verify_answer("GAS_PER_BLOB is 2**17 ​[eip-4844#003].", CTX)
    assert graded_grounding(res) == 0.0
    assert graded_grounding(verify_answer("", CTX)) == 0.0


def test_v1_breakdown_reports_hallucinated_citations():
    _, bd = reward_v1("X is 1 [eip-4844#999].", ex())
    assert bd["hallucinated_citations"] == 1 and bd["verified"] is False


# --- v2 terms ---


def test_v2_oracle_scores_one():
    for e in (ex(), multi(), comparison()):
        assert reward_v2(e.task.gold_answer, e)[0] == pytest.approx(1.0, abs=0.005)


def test_correctness_rewards_recall_and_quarters_the_score_per_extra_claim():
    e = ex()
    assert correctness("FIELD_ELEMENTS_PER_BLOB is 4096", e, 0.25)[0] == 1.0
    assert correctness("FIELD_ELEMENTS_PER_BLOB is 8192", e, 0.25)[0] == 0.0
    c, recall, precision, extras = correctness(
        "FIELD_ELEMENTS_PER_BLOB is 4096 and 786432", e, 0.25)
    assert (recall, precision, extras, c) == (1.0, 0.5, 1, 0.25)
    c, *_, extras = correctness("is 4096 and 786432 and 2400 and 7", e, 0.25)
    assert extras == 3 and c == pytest.approx(0.25**3)


def test_correctness_first_design_is_recall_times_precision():
    c, recall, precision, extras = correctness("is 4096 and 786432", ex(), None)
    assert (c, recall, precision, extras) == (0.5, 1.0, 0.5, 1)


def test_correctness_treats_question_and_eip_tokens_as_neutral():
    c, recall, precision, extras = correctness("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096", ex(),
                                               0.25)
    assert (recall, precision, extras) == (1.0, 1.0, 0)


def test_correctness_accepts_equivalent_spellings_in_both_directions():
    e = ex(gold_tokens=["0x40"])
    assert correctness("the byte is 64", e, 0.25)[0] == 1.0
    assert correctness("the byte is 0x40", e, 0.25)[0] == 1.0
    small = ex(gold_tokens=["0x2"])
    assert correctness("type 2", small, 0.25)[0] == 1.0
    assert correctness("type 0x02", small)[0] == 1.0
    assert correctness("type 2", ex(gold_tokens=["2"]), 0.25)[0] == 1.0


def test_correctness_with_no_gold_tokens_is_neutral():
    assert correctness("anything 42", ex(gold_tokens=[])) == (1.0, 1.0, 1.0, 0)


def test_correctness_of_an_answer_without_claims_is_zero():
    assert correctness("It is large.", ex())[0] == 0.0


def test_multi_fact_one_wrong_value_loses_recall_and_pays_for_the_extra():
    e = multi()
    c, recall, precision, extras = correctness(
        "FIELD_ELEMENTS_PER_BLOB is 4096 and GAS_PER_BLOB is 5", e, 0.25)
    assert recall == 0.5 and extras == 1 and c == pytest.approx(0.125)


def test_citation_quality_is_precision_times_recall_over_gold_groups():
    e = comparison()
    assert citation_quality(["eip-1559#011", "eip-4844#003"], e)[0] == 1.0
    assert citation_quality(["eip-4844#003"], e) == (0.5, 1.0, 0.5)
    three = citation_quality(["eip-4844#003", "eip-1559#011", "eip-2930#002"], e)
    assert three[0] == pytest.approx(2 / 3)
    assert citation_quality([], e)[0] == 0.0


def test_citation_quality_is_neutral_without_gold_groups():
    assert citation_quality(["x"], unanswerable()) == (1.0, 1.0, 1.0)


def test_relevance_counts_repeated_question_tokens_excluding_eip_numbers():
    e = ex()
    assert relevance("FIELD_ELEMENTS_PER_BLOB is 4096", e) == 1.0
    assert relevance("It is 4096", e) == 0.0
    assert relevance("anything", ex(question="What is it in EIP-4844?")) == 1.0


def test_relevance_ignores_the_quoted_section_title():
    e = ex(question='In EIP-4844 (section "CALLF"), what fills _____ in "uses 7 items"?')
    assert relevance("uses 7 items", e) == 1.0


def test_length_factor_is_flat_to_the_budget_then_falls_to_zero():
    e = ex()  # gold has 7 words -> budget 1.5 * 7 + 15 = 25.5
    cfg = V2Config()
    assert length_factor("word " * 10, e, cfg)[0] == 1.0
    mid = length_factor("word " * 38, e, cfg)[0]
    assert 0.0 < mid < 1.0
    assert length_factor("word " * 80, e, cfg)[0] == 0.0
    _, words, budget = length_factor("one two three", e, cfg)
    assert words == 3 and budget == pytest.approx(1.5 * len(GOLD.split()) - 1.5 + 15, abs=2)


def test_redundancy_factor_penalises_repeated_sentences_and_citation_markers():
    assert redundancy_factor("A is 1 [eip-1#001]. B is 2 [eip-1#001].")[0] == 1.0
    f, dup_s, _ = redundancy_factor("A is 1 [eip-1#001]. A is 1 [eip-1#001]. A is 1 [eip-1#001].")
    assert dup_s == pytest.approx(2 / 3) and f == pytest.approx(1 / 3)
    f, _, dup_c = redundancy_factor("A is 1 [eip-1#001][eip-1#001][eip-1#001][eip-1#001].")
    assert dup_c == 0.75 and f == 0.25
    assert redundancy_factor("") == (1.0, 0.0, 0.0)


def test_redundancy_ignores_case_and_punctuation_when_matching_sentences():
    assert redundancy_factor("A is 1 [eip-1#001]. a is 1! [eip-1#001]")[0] < 1.0


# --- v2 end to end ---


def test_v2_abstention_calibration():
    assert reward_v2(ABSTAIN_TEXT, unanswerable())[0] == 1.0
    assert reward_v2(ABSTAIN_TEXT, ex())[0] == 0.0
    assert reward_v2("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003].", unanswerable())[0] == 0.0
    assert reward_v2("", unanswerable())[0] == 0.0
    _, bd = reward_v2(ABSTAIN_TEXT, unanswerable())
    assert bd["abstained"] is True and bd["score_reason"] == "abstained"
    _, bd = reward_v2("I think 2 [eip-1559#011].", unanswerable())
    assert bd["score_reason"] == "answered_unanswerable"


def test_v2_abstention_must_be_the_entire_answer():
    sneaky = ABSTAIN_TEXT + " FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]."
    assert reward_v2(sneaky, unanswerable())[0] == 0.0


def test_v2_wrong_value_scores_only_the_floor_or_less():
    s, bd = reward_v2("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 8192 [eip-4844#003].", ex())
    assert s < 0.12 and bd["correctness"] == 0.0


def test_v2_floor_gives_a_gradient_to_well_formed_but_wrong_answers():
    s_wrong, _ = reward_v2("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 786432 [eip-4844#003].", ex())
    s_uncited, _ = reward_v2("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096.", ex())
    assert 0.0 < s_wrong <= 0.1 + 1e-9
    assert s_uncited < s_wrong


def test_v2_ablation_switches_change_the_score_as_documented():
    cfg = V2Config()
    spam = " ".join([GOLD] * 8)
    full, _ = reward_v2(spam, ex(), cfg)
    no_red, _ = reward_v2(spam, ex(), cfg.without("redundancy"))
    no_len, _ = reward_v2(spam, ex(), cfg.without("length"))
    assert full < 0.05 and no_red < full + 0.5 and no_len > full
    assert reward_v2(spam, ex(), cfg.without("redundancy").without("length"))[0] > 0.9

    cite_all = ("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 "
                "[eip-4844#003][eip-1559#011][eip-2930#002].")
    assert reward_v2(cite_all, ex(), cfg)[0] < 0.5
    assert reward_v2(cite_all, ex(), cfg.without("citation"))[0] > 0.9

    wrong_chunk = "In EIP-1559, ELASTICITY_MULTIPLIER is 2 [eip-1559#011]."
    assert reward_v2(wrong_chunk, ex(), cfg)[0] <= cfg.floor + 1e-9  # grounded but wrong
    assert reward_v2(wrong_chunk, ex(), cfg.without("correctness").without("citation"))[0] > 0.6

    assert reward_v2(ABSTAIN_TEXT, unanswerable(), cfg.without("abstention"))[0] == 0.0
    hallucinated = "In EIP-1559, ELASTICITY_MULTIPLIER is 2 [eip-1559#011]."
    assert reward_v2(hallucinated, unanswerable(), cfg.without("abstention"))[0] > 0.5
    assert reward_v2(hallucinated, unanswerable(), cfg)[0] == 0.0


def test_v2_config_without_rejects_unknown_terms():
    with pytest.raises(ValueError, match="unknown reward term"):
        V2Config().without("vibes")
    assert set(V2_TERMS) == {"abstention", "correctness", "citation", "relevance", "length",
                             "redundancy", "consistency"}


def test_v2_breakdown_has_every_term():
    _, bd = reward_v2(GOLD, ex())
    for key in ("grounded", "correctness", "citation_quality", "relevance", "length_factor",
                "redundancy_factor", "words", "verified", "hallucinated_citations"):
        assert key in bd


def test_score_completion_dispatches_and_validates_version():
    assert score_completion("v0", GOLD, ex())[0] == 1.0
    assert score_completion("v1", GOLD, ex())[1]["version"] == "v1"
    assert score_completion("v2", GOLD, ex(), V2Config(floor=0.5))[1]["version"] == "v2"
    with pytest.raises(ValueError, match="unknown reward version"):
        score_completion("v9", GOLD, ex())


def test_scores_are_always_in_unit_interval_on_random_text():
    rng = random.Random(0)
    alphabet = string.ascii_letters + string.digits + " []#-_.*`\n​"
    pieces = ["[eip-4844#003]", "[eip-1559#011]", "4096", "0x03", "2**17",
              "FIELD_ELEMENTS_PER_BLOB", ABSTAIN_TEXT, "```", "- ", "<think>", "</think>"]
    for _ in range(300):
        parts = [rng.choice(pieces) if rng.random() < 0.4 else
                 "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
                 for _ in range(rng.randint(0, 14))]
        text = " ".join(parts)
        for e in (ex(), unanswerable(), multi(), comparison()):
            for v in ("v0", "v1", "v2"):
                s, _ = score_completion(v, text, e)
                assert 0.0 <= s <= 1.0, (v, text)


def test_rewards_are_deterministic():
    text = "In EIP-4844, GAS_PER_BLOB is 2**17 [eip-4844#003]."
    assert reward_v2(text, multi()) == reward_v2(text, multi())


# --- TRL wrappers ---


def test_trl_wrapper_signature_name_and_scores():
    fn = make_trl_reward("v2")
    assert fn.__name__ == "groundrl_reward_v2"
    blobs = [ex().to_json(), unanswerable().to_json()]
    out = fn(prompts=["p", "p"], completions=[GOLD, ABSTAIN_TEXT], example_json=blobs)
    assert out == pytest.approx([1.0, 1.0], abs=0.005) and all(isinstance(x, float) for x in out)


def test_trl_wrapper_accepts_chat_style_completions_and_extra_kwargs():
    fn = make_trl_reward("v0")
    comps = [[{"role": "assistant", "content": f"<think>x</think>{GOLD}"}]]
    out = fn(["p"], comps, example_json=[ex().to_json()], completion_ids=[[1]], trainer_state=None)
    assert out == [1.0]


def test_trl_wrapper_reports_diagnostics_through_log_metric():
    logged: dict[str, float] = {}
    fn = make_trl_reward("v2")
    fn(["p", "p"], [GOLD, "FIELD_ELEMENTS_PER_BLOB is 4096 [eip-9#999]."],
       example_json=[ex().to_json()] * 2, log_metric=lambda k, v: logged.__setitem__(k, v))
    assert logged["groundrl/verified_rate"] == 0.5
    assert logged["groundrl/hallucinated_citation_rate"] == 0.5
    assert {"groundrl/abstain_rate", "groundrl/mean_words", "groundrl/correctness"} <= set(logged)


def test_trl_wrapper_errors_are_explicit():
    fn = make_trl_reward("v1")
    with pytest.raises(KeyError, match="example_json"):
        fn(["p"], ["c"])
    with pytest.raises(ValueError, match="one example per completion"):
        fn(["p"], ["c", "d"], example_json=[ex().to_json()])
    with pytest.raises(ValueError, match="unknown reward version"):
        make_trl_reward("v7")


def test_trl_wrapper_custom_column_and_config():
    fn = make_trl_reward("v2", V2Config(floor=0.5), column="blob")
    out = fn(["p"], [GOLD], blob=[ex().to_json()])
    assert out[0] == pytest.approx(1.0, abs=0.005)


def test_reward_module_does_not_import_trl_or_torch():
    # In a child interpreter: sys.modules is session-global, so checking it here
    # would silently stop testing anything once another test imports torch.
    source = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {src!r})
        import groundrl.rewards  # noqa: F401
        heavy = [m for m in ("trl", "torch", "transformers", "peft", "datasets")
                 if m in sys.modules]
        print(",".join(heavy))
        """
    ).format(src=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", f"groundrl.rewards imported: {r.stdout.strip()}"


def test_example_json_roundtrip():
    e = multi()
    again = Example.from_json(e.to_json())
    assert again.task == e.task and again.context == e.context
    assert again.context_ids == e.context_ids


# --- the committed dataset -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_examples(dataset):
    from groundrl.example import make_examples

    return {s: make_examples(v, dataset.index) for s, v in dataset.tasks.items()}


def test_every_oracle_answer_passes_v0_and_scores_near_one_under_v2(real_examples):
    for split, examples in real_examples.items():
        for e in (e for e in examples if e.task.answerable):
            assert reward_v0(e.task.gold_answer, e)[0] == 1.0, e.task.id
            assert reward_v2(e.task.gold_answer, e)[0] >= 0.97, (split, e.task.id)


def test_every_unanswerable_task_rewards_abstention_under_v2_only(real_examples):
    for examples in real_examples.values():
        for e in (e for e in examples if not e.task.answerable):
            assert reward_v2(ABSTAIN_TEXT, e)[0] == 1.0
            assert reward_v0(ABSTAIN_TEXT, e)[0] == 0.0
            assert reward_v1(ABSTAIN_TEXT, e)[0] == 0.0


def test_examples_preserve_context_order(real_examples, dataset):
    for e in real_examples["test"][:50]:
        assert e.context_ids == e.task.context_ids
        assert all(dataset.index.by_id[c].text == t for c, t in e.context.items())


# --- consistency (answering twice) and revisions -------------------------------------------------


def test_contradictions_flags_the_same_sentence_with_different_numbers():
    e = ex()
    twice = ("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]. "
             "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 786432 [eip-4844#003].")
    assert contradictions(twice, e) == 1
    same = ("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]. "
            "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 0x1000 [eip-4844#003].")
    assert contradictions(same, e) == 0  # equivalent spellings agree


def test_contradictions_does_not_flag_legitimate_multi_fact_answers():
    assert contradictions(multi().task.gold_answer, multi()) == 0
    cmp_ = ex(eips=["eip-198", "eip-1167"])
    dates = ("EIP-198 was created on 2017-01-30 [eip-198#000]. "
             "EIP-1167 was created on 2018-06-22 [eip-1167#000].")
    assert contradictions(dates, cmp_) == 0  # different EIPs are different sentences
    siblings = ("a single G1 point (128 bytes) [eip-2537#018]. "
                "a single G2 point (256 bytes) [eip-2537#018].")
    assert contradictions(siblings, ex()) == 0


def test_contradictions_ignores_sentences_without_numbers():
    assert contradictions("It is large [eip-4844#003]. It is large [eip-4844#003].", ex()) == 0
    assert contradictions("", ex()) == 0


def test_v2_penalises_answering_twice_but_rev1_did_not_fully():
    twice = ("In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]. "
             "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 786432 [eip-4844#003].")
    final, bd = reward_v2(twice, ex())
    assert bd["contradictions"] == 1 and bd["consistency_factor"] == 0.25
    no_cons, _ = reward_v2(twice, ex(), V2Config().without("consistency"))
    rev1, _ = reward_v2(twice, ex(), V2Config.rev1())
    assert final < no_cons <= rev1 + 1e-9 or final < rev1
    assert final < 0.2


def test_rev1_preset_has_no_consistency_term_and_the_precision_formula():
    cfg = V2Config.rev1()
    assert cfg.consistency is False and cfg.extra_claim_factor is None
    assert V2Config().consistency is True and V2Config().extra_claim_factor == 0.25


def test_inline_hedge_pays_for_the_extra_claim():
    hedged = "In EIP-4844, FIELD_ELEMENTS_PER_BLOB is 4096 (or 786432) [eip-4844#003]."
    s, bd = reward_v2(hedged, ex())
    assert bd["extra_claims"] == 1 and s < 0.4
