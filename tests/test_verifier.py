from __future__ import annotations

import json

import pytest

from groundrl.verifier import SentenceReport, verify_answer

BLOB = ("EIP-4844: Shard Blob Transactions / Parameters\n"
        "| Constant | Value |\n| - | - |\n"
        "| `BLOB_TX_TYPE` | `Bytes1(0x03)` |\n| `FIELD_ELEMENTS_PER_BLOB` | `4096` |\n"
        "| `GAS_PER_BLOB` | `2**17` |\n| `MAX_BLOB_GAS_PER_BLOCK` | `786432` |")
FEE = ("EIP-1559: Fee market change / Specification\n"
       "INITIAL_BASE_FEE = 1000000000\nELASTICITY_MULTIPLIER = 2\n"
       "The BASEFEE opcode is 0x48 and 58 is the other value. Stake is 32 ETH.")
CHUNKS = {"eip-4844#003": BLOB, "eip-1559#011": FEE}


def check(answer, ids=None):
    return verify_answer(answer, CHUNKS, ids)


def test_correct_grounded_answer_passes_with_provenance():
    res = check("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003].")
    assert res.ok and res.issues == []
    assert res.provenance() == {"eip-4844#003": ["4096", "field_elements_per_blob"]}
    assert res.support_ratio == 1.0 and res.cited_chunks == ["eip-4844#003"]


def test_wrong_number_is_rejected():
    res = check("FIELD_ELEMENTS_PER_BLOB is 8192 [eip-4844#003].")
    assert not res.ok and "unsupported claim '8192'" in res.issues[0]
    assert res.support_ratio == 0.5  # the identifier was fine, the number was not


def test_uncited_sentence_is_rejected():
    res = check("FIELD_ELEMENTS_PER_BLOB is 4096.")
    assert not res.ok
    assert any("uncited sentence" in i for i in res.issues)


def test_one_uncited_sentence_among_cited_ones_fails_the_answer():
    res = check("GAS_PER_BLOB is 2**17 [eip-4844#003]. It is a nice number.")
    assert not res.ok and [s.passed for s in res.sentences] == [True, False]


@pytest.mark.parametrize("answer", ["", "   ", "\n\n", "```\n```"])
def test_empty_answers_never_pass(answer):
    res = check(answer)
    assert not res.ok and res.issues == ["empty answer"]


def test_citation_only_answer_has_no_content_and_fails():
    res = check("[eip-4844#003]")
    assert not res.ok and any("no content besides citations" in i for i in res.issues)


def test_spoofed_citation_to_a_chunk_that_does_not_exist_is_rejected():
    res = check("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#999].")
    assert not res.ok
    assert res.invalid_citations == ["eip-4844#999"]
    assert any("not in the provided context" in i for i in res.issues)


def test_citation_to_a_real_chunk_outside_the_context_is_rejected():
    res = check("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003].", ids=["eip-1559#011"])
    assert not res.ok and res.invalid_citations == ["eip-4844#003"]


def test_context_ids_default_to_all_chunks_and_unknown_ids_are_ignored():
    assert check("ELASTICITY_MULTIPLIER is 2 [eip-1559#011].").ok
    res = verify_answer("ELASTICITY_MULTIPLIER is 2 [eip-1559#011].", CHUNKS,
                        ["eip-1559#011", "zzz"])
    assert res.ok


def test_one_valid_and_one_invalid_citation_fails_but_records_both():
    res = check("GAS_PER_BLOB is 2**17 [eip-4844#003][eip-1559#999].")
    assert not res.ok and res.cited_chunks == ["eip-4844#003"]
    assert res.sentences[0].invalid == ["eip-1559#999"]


def test_multi_citation_sentence_supports_each_token_from_the_right_chunk():
    res = check("BLOB_TX_TYPE is 0x03 and INITIAL_BASE_FEE is 1000000000 "
                "[eip-4844#003][eip-1559#011].")
    assert res.ok
    prov = res.provenance()
    assert "0x3" in prov["eip-4844#003"] and "1000000000" in prov["eip-1559#011"]


def test_token_cited_to_the_wrong_chunk_is_rejected():
    res = check("INITIAL_BASE_FEE is 1000000000 [eip-4844#003].")
    assert not res.ok and any("1000000000" in i for i in res.issues)


def test_trailing_citation_after_the_period_is_accepted():
    assert check("FIELD_ELEMENTS_PER_BLOB is 4096. [eip-4844#003]").ok


def test_citation_before_period_and_mid_sentence_are_accepted():
    assert check("The value [eip-4844#003] of FIELD_ELEMENTS_PER_BLOB is 4096.").ok


def test_markdown_fence_wrapper_does_not_hide_claims():
    good = "```text\nFIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003].\n```"
    bad = "```text\nFIELD_ELEMENTS_PER_BLOB is 9999 [eip-4844#003].\n```"
    assert check(good).ok
    assert not check(bad).ok


def test_inline_code_and_bold_markup_are_transparent():
    assert check("**FIELD_ELEMENTS_PER_BLOB** is `4096` [eip-4844#003].").ok


def test_bulleted_and_numbered_answers_are_checked_line_by_line():
    ok = "- BLOB_TX_TYPE is 0x03 [eip-4844#003]\n2. GAS_PER_BLOB is 2**17 [eip-4844#003]"
    assert check(ok).ok
    wrong = "- BLOB_TX_TYPE is 0x03 [eip-4844#003]\n- GAS_PER_BLOB is 2**18 [eip-4844#003]"
    assert not check(wrong).ok


def test_hex_and_decimal_are_equivalent_spellings():
    res = check("The BASEFEE opcode is 72 [eip-1559#011].")
    assert res.ok
    assert [c.kind for c in res.claims if c.token == "72"] == ["hex-dec"]


def test_power_of_two_and_value_are_equivalent_spellings():
    res = check("GAS_PER_BLOB is 131072 [eip-4844#003].")
    assert res.ok and any(c.kind == "pow2" for c in res.claims)
    assert not check("The cap is 2**10 [eip-4844#003].").ok


def test_eth_amounts_match_their_gwei_spelling():
    assert check("The stake is 32000000000 gwei [eip-1559#011].").ok
    assert not check("The stake is 33000000000 gwei [eip-1559#011].").ok


def test_substring_numbers_do_not_count():
    # "58" is in the chunk; "8" must not be found inside it, and 5 not inside 58.
    assert check("The other value is 58 [eip-1559#011].").ok
    assert not check("The other value is 8 [eip-1559#011].").ok
    assert not check("The other value is 5 [eip-1559#011].").ok


def test_identifier_must_match_exactly_including_underscores():
    assert not check("GASPERBLOB is 2**17 [eip-4844#003].").ok
    assert not check("GAS_PER_BLOB_X is 2**17 [eip-4844#003].").ok


def test_lower_case_identifiers_are_not_claims():
    res = check("gas_per_blob is 2**17 [eip-4844#003].")
    assert res.ok and [c.token for c in res.claims] == ["2**17"]


@pytest.mark.parametrize("answer", [
    "FIELD_ELEMENTS_PER_BLOB is ４０９６ [eip-4844#003].",       # fullwidth digits
    "FIELD_ELEMENTS_PER_BLOB is ٤٠٩٦ [eip-4844#003].",       # Arabic-Indic digits
    "FIELD_ELEMENTS_PER_BLOB is 40​96 [eip-4844#003].",                     # zero-width space
    "FIELD_ELEMENTS_PER_BLOB is 4096 [eip‑4844#003].",                      # non-breaking hyphen
    "FIELD_ELEMENTS_PER_BLOB is 4096 ［eip-4844#003］.",                  # fullwidth brackets
    "FIELD_ELEMENTS_PER_BLOB is 4096‮ [eip-4844#003].",                     # bidi override
])
def test_unicode_lookalikes_and_invisible_characters_are_rejected(answer):
    res = check(answer)
    assert not res.ok and res.suspicious


def test_cyrillic_lookalike_identifier_cannot_hide_a_claim():
    # Latin "M" and "A" replaced by Cyrillic: the identifier regex would not see it.
    res = check("МАX_BLOB_GAS_PER_BLOCK is 786432 [eip-4844#003].")
    assert not res.ok and res.suspicious


def test_superscript_digits_fold_to_the_real_number_and_are_flagged():
    res = check("GAS_PER_BLOB is 2¹⁷ [eip-4844#003].")
    assert not res.ok  # flagged as suspicious; folding alone would otherwise verify "217"


def test_typographic_punctuation_is_allowed():
    assert check("FIELD_ELEMENTS_PER_BLOB — “is” 4096 [eip-4844#003].").ok


@pytest.mark.parametrize("bad,kind", [
    ("[eip-4844#3]", "no zero padding"), ("[EIP-4844#003]", "upper case"),
    ("(eip-4844#003)", "parentheses"), ("[eip 4844#003]", "space"),
])
def test_malformed_citations_do_not_count_and_are_counted_as_near_misses(bad, kind):
    res = check(f"FIELD_ELEMENTS_PER_BLOB is 4096 {bad}.")
    assert not res.ok, kind
    assert res.near_miss == 1 and any("malformed citation" in i for i in res.issues)


def test_numbers_inside_citations_are_not_claims():
    res = check("GAS_PER_BLOB is 2**17 [eip-4844#003].")
    assert all("003" not in c.token for c in res.claims)


def test_paraphrase_without_checkable_tokens_passes_vacuously_a_known_blind_spot():
    res = check("This proposal changes how blobs are priced [eip-4844#003].")
    assert res.ok and res.checked_tokens == 0 and res.support_ratio == 1.0


def test_right_token_in_a_wrong_claim_passes_a_known_blind_spot():
    # 786432 is the max blob gas per block, not FIELD_ELEMENTS_PER_BLOB, but both tokens occur.
    res = check("FIELD_ELEMENTS_PER_BLOB is 786432 [eip-4844#003].")
    assert res.ok


def test_coverage_is_the_share_of_content_words_found_in_cited_chunks():
    full = check("Shard Blob Transactions parameters [eip-4844#003].")
    none = check("Totally unrelated words appear here [eip-4844#003].")
    assert full.sentences[0].coverage > 0.9 and none.sentences[0].coverage < 0.2


def test_to_dict_is_json_serialisable_and_complete():
    res = check("FIELD_ELEMENTS_PER_BLOB is 4096 [eip-4844#003]. Nothing else.")
    blob = json.loads(json.dumps(res.to_dict()))
    assert blob["ok"] is False and blob["checked_tokens"] == 2
    assert blob["sentences"][0]["valid"] == ["eip-4844#003"]
    assert blob["provenance"]["eip-4844#003"]


def test_cited_chunks_keep_first_use_order_and_deduplicate():
    res = check("GAS_PER_BLOB is 2**17 [eip-4844#003]. ELASTICITY_MULTIPLIER is 2 [eip-1559#011]. "
                "BLOB_TX_TYPE is 0x03 [eip-4844#003].")
    assert res.cited_chunks == ["eip-4844#003", "eip-1559#011"]


def test_sentence_report_properties():
    s = SentenceReport(0, "x", [], [], [], checked=0, supported=0)
    assert not s.passed and s.support_ratio == 1.0
    s2 = SentenceReport(0, "x", ["a"], ["a"], [], checked=2, supported=1)
    assert not s2.passed and s2.support_ratio == 0.5


def test_unsupported_claim_message_names_the_cited_chunks():
    res = check("GAS_PER_BLOB is 5 [eip-1559#011].")
    assert any("eip-1559#011" in i for i in res.issues)


def test_control_characters_are_rejected():
    assert not check("GAS_PER_BLOB is 2**17 \x07 [eip-4844#003].").ok
