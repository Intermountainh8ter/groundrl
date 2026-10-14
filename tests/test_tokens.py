from __future__ import annotations

import pytest

from groundrl.protocol import (
    ABSTAIN_TEXT,
    CITATION,
    LOOSE_CITATION,
    abstain_key,
    is_abstention,
)
from groundrl.tokens import (
    claim_tokens,
    content_words,
    has_negation,
    identifier_tokens,
    is_numeric_token,
    norm_number,
    normalize_text,
    number_variants,
    numeric_tokens,
    sentences,
    stem,
    strip_citations,
    suspicious_chars,
    unit_variants,
    word_count,
)


def test_numbers_hex_and_powers_are_claim_tokens():
    toks = claim_tokens("Set 4096 and 0x03 and 2**17 and 1,000,000 and 1_000.")
    assert {"4096", "0x3", "2**17", "1000000", "1000"} <= toks


def test_identifiers_are_case_folded_and_keep_underscores():
    assert identifier_tokens("FIELD_ELEMENTS_PER_BLOB and BLOB_TX_TYPE") == {
        "field_elements_per_blob", "blob_tx_type"}
    assert "gasperblob" not in claim_tokens("GAS_PER_BLOB")


def test_rfc2119_words_and_acronyms_are_not_claims():
    assert identifier_tokens("Clients MUST NOT use the EVM or RLP for ERC and ETH.") == set()


def test_opcode_mnemonics_with_digits_are_identifiers():
    assert "push0" in identifier_tokens("The PUSH0 instruction")


def test_substring_numbers_are_separate_tokens():
    assert numeric_tokens("58") == {"58"}
    assert "8" not in numeric_tokens("58")


def test_decimal_with_thousands_groups_is_one_token_but_lists_are_not_merged():
    assert numeric_tokens("1,2,3") == {"1", "2", "3"}
    assert numeric_tokens("12,345") == {"12345"}


def test_citations_are_not_claim_tokens():
    assert claim_tokens("Value [eip-4844#003].") == set()
    assert strip_citations("a [eip-1#001] b") == "a  b"


def test_norm_number_canonical_forms():
    assert norm_number("0x0B") == "0xb"
    assert norm_number("1,000") == "1000"
    assert norm_number("2 ^ 5") == "2**5"
    assert norm_number("0x00") == "0x0"


def test_number_variants_hex_decimal_and_powers():
    assert "72" in number_variants("0x48")
    assert "0x48" in number_variants("72")
    assert "4096" in number_variants("2**12")
    assert "2**12" in number_variants("4096")
    assert number_variants("abc") == {"abc"}
    assert number_variants("3") == {"3"}  # too small to alias to hex


def test_unit_variants_cover_eth_gwei_and_wei():
    v = unit_variants("stake 32 ETH and 5 gwei")
    assert "32000000000000000000" in v["32"] and "32000000000" in v["32"]
    assert v["5"] == {"5000000000"}


def test_fullwidth_and_superscript_digits_fold_to_ascii():
    assert normalize_text("４０９６") == "4096"
    assert numeric_tokens("value ４０９６") == {"4096"}
    assert numeric_tokens("2¹⁷")  # superscripts become digits, still a token


@pytest.mark.parametrize("text,flag", [
    ("plain ascii 4096", False),
    ("curly ’ quotes — dash ≥ symbols", False),
    ("zero​width", True),
    ("arabic ٤٠٩٦", True),
    ("cyrillic МАX_BLOBS", True),
    ("greek ο", True),
    ("non-breaking hyphen eip‑1559", True),
    ("fullwidth bracket ［eip-1#001］", True),
    ("bidi ‮ override", True),
    ("control \x07 bell", True),
    ("tabs\tand\nnewlines\r are fine", False),
])
def test_suspicious_characters(text, flag):
    assert bool(suspicious_chars(text)) is flag


def test_suspicious_chars_reports_code_points():
    assert suspicious_chars("a​b​") == ["U+200B"]


def test_sentences_split_on_boundaries_and_newlines():
    got = sentences("First is 4096. Second is 0x03! Third?\nFourth line [eip-1#001].")
    assert got == ["First is 4096.", "Second is 0x03!", "Third?", "Fourth line [eip-1#001]."]


def test_decimals_and_versions_do_not_split_sentences():
    assert sentences("The ratio is 1.5 and v2.0 holds.") == ["The ratio is 1.5 and v2.0 holds."]


def test_trailing_citation_after_period_attaches_to_previous_sentence():
    got = sentences("The value is 4096. [eip-4844#003]")
    assert got == ["The value is 4096. [eip-4844#003]"]


def test_citation_only_first_line_stays_its_own_fragment():
    assert sentences("[eip-4844#003]") == ["[eip-4844#003]"]


def test_list_markers_and_fences_are_stripped_but_content_is_kept():
    text = "```text\n- A is 1 [eip-1#001].\n2. B is 2 [eip-1#001].\n```"
    assert sentences(text) == ["A is 1 [eip-1#001].", "B is 2 [eip-1#001]."]


def test_blank_lines_are_ignored():
    assert sentences("\n\n  \nOnly this.\n") == ["Only this."]


def test_content_words_stem_and_drop_stopwords():
    words = content_words("The blobs are carried by transactions [eip-4844#003]")
    assert words == {"blob", "carr", "transaction"}
    assert content_words("of the and") == set()


def test_stem_leaves_short_words_alone():
    assert stem("gas") == "gas" and stem("fees") == "fees"  # four letters or fewer: untouched
    assert stem("blobs") == "blob" and stem("running") == "runn"


def test_negation_detection():
    assert has_negation("This does not burn")
    assert has_negation("It isn't removed")
    assert not has_negation("It burns the fee")


def test_word_count_ignores_citations():
    assert word_count("one two [eip-1#001] three") == 3


def test_is_numeric_token():
    assert is_numeric_token("4096") and is_numeric_token("0x3")
    assert not is_numeric_token("blob") and not is_numeric_token("")


def test_citation_regex_is_strict():
    assert CITATION.findall("[eip-1559#007] [eip-1559#7] [EIP-1559#007] (eip-1559#007)") == [
        "eip-1559#007"]
    assert CITATION.fullmatch("[eip-1559#1000]")


def test_loose_citation_regex_flags_format_drift():
    for bad in ("[eip-1559#7]", "[EIP-1559#007]", "(eip-1559#007)", "[eip 1559#007]",
                "[eip-1559:007]", "[ERC-20#003]"):
        assert LOOSE_CITATION.search(bad), bad
    assert not LOOSE_CITATION.search("nothing to see")


def test_abstention_is_strict_about_content_but_lenient_about_form():
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("  i CANNOT answer this from the provided context!!  ")
    assert not is_abstention(ABSTAIN_TEXT + " But it is probably 4096 [eip-1#001].")
    assert not is_abstention("")
    assert not is_abstention("I do not know.")
    assert abstain_key("A-b_c") == "a b c"
