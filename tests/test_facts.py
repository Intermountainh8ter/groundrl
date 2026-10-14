from __future__ import annotations

from collections import Counter

import pytest

from groundrl.chunker import Chunk, chunk_document
from groundrl.facts import (
    BLANK,
    CLOZE,
    CONSTANT,
    METADATA,
    OPCODE,
    TYPE_BYTE,
    Fact,
    cloze_candidates,
    extract_facts,
    paragraph_sentences,
    parse_value,
    plain,
    supports,
    token_supported,
)
from groundrl.tokens import claim_tokens
from tests.helpers import make_chunks, make_doc


def mk(text: str, doc="eip-900", section="Specification", kind="prose", idx=1) -> Chunk:
    return Chunk(f"{doc}#{idx:03d}", doc, int(doc.split("-")[1]), "T", section, kind,
                 f"EIP-900: T / {section}\n{text}")


@pytest.mark.parametrize("cell,expected", [
    ("`4096`", "4096"), ("`Bytes1(0x03)`", "0x03"), ("0x0b", "0x0b"), ("`2**17`", "2**17"),
    ("`uint64(2**14)` (= 16,384)", "2**14"), ("`Bytes1('0x02')`", "0x02"), ("1,000", "1,000"),
    ("2_097_152", "2_097_152"), ("TBD", None), ("`2 * MAX_CODE_SIZE`", None),
    ("four", None), ("", None), ("`DomainType('0x0B000000')`", "0x0B000000"),
])
def test_parse_value(cell, expected):
    assert parse_value(cell) == expected


def test_table_rows_become_constant_facts():
    chunk = mk("| Constant | Value |\n| - | - |\n| `MAX_THING` | `64` |\n| `OTHER_KEY` | TBD |")
    facts = extract_facts([chunk])
    named = {f.name: f for f in facts if f.kind == CONSTANT}
    assert set(named) == {"MAX_THING"}
    assert named["MAX_THING"].value == "64" and named["MAX_THING"].tokens == ("64",)
    assert named["MAX_THING"].chunk_id == "eip-900#001"


def test_three_column_tables_use_the_value_column():
    chunk = mk("| Name | Type | Value |\n| - | - | - |\n| `SOME_LIMIT` | uint64 | `7` |")
    assert [(f.name, f.value) for f in extract_facts([chunk])] == [("SOME_LIMIT", "7")]


def test_assignment_lines_and_inline_assignments():
    chunk = mk("INITIAL_BASE_FEE = 1000000000\nMAX_X: 5 not matched\n"
               "Define a constant `LEN_PER_PAIR = 384`; done.")
    got = {f.name: f.value for f in extract_facts([chunk]) if f.kind != CLOZE}
    assert got == {"INITIAL_BASE_FEE": "1000000000", "LEN_PER_PAIR": "384"}


def test_name_with_conflicting_values_is_dropped_as_ambiguous():
    a = mk("| `LIMIT_Q` | `1` |\n| - | - |", idx=1)
    b = mk("LIMIT_Q = 2", idx=2)
    assert [f for f in extract_facts([a, b]) if f.name == "LIMIT_Q"] == []


def test_same_value_in_two_chunks_is_kept_once():
    a = mk("LIMIT_Q = 33", idx=1)
    b = mk("LIMIT_Q = 33", idx=2)
    facts = [f for f in extract_facts([a, b]) if f.name == "LIMIT_Q"]
    assert len(facts) == 1 and facts[0].chunk_id == "eip-900#001"


def test_opcode_prose_patterns():
    chunk = mk("Add a `BASEFEE` opcode at `(0x48)`.\nThe `PUSH0` (`0x5f`) instruction is new.\n"
               "`DELEGATECALL`: `0xf4`, takes operands.\nA new opcode `AUTH` shall be at `0xf6`.")
    got = {f.name: f.value for f in extract_facts([chunk]) if f.kind == OPCODE}
    assert got == {"BASEFEE": "0x48", "PUSH0": "0x5f", "DELEGATECALL": "0xf4", "AUTH": "0xf6"}


def test_opcode_extraction_requires_evm_context_and_skips_noise():
    chunk = mk("The RLP encoding 0xc0 is an empty list. ABI selector 0xab.")
    assert [f for f in extract_facts([chunk]) if f.kind == OPCODE] == []


def test_type_byte_and_opcode_named_constants_are_classified_by_hex_value():
    chunk = mk("| `BLOB_TX_TYPE` | `Bytes1(0x03)` |\n| - | - |\n| `HASH_OPCODE_BYTE` | `0x49` |\n"
               "| `HASH_OPCODE_GAS` | `3` |\n| `DEPOSIT_REQUEST_TYPE` | `b'0'` |")
    kinds = {f.name: f.kind for f in extract_facts([chunk])}
    assert kinds["BLOB_TX_TYPE"] == TYPE_BYTE
    assert kinds["HASH_OPCODE_BYTE"] == OPCODE
    assert kinds["HASH_OPCODE_GAS"] == CONSTANT  # a gas cost, not an opcode byte
    assert kinds["DEPOSIT_REQUEST_TYPE"] == CONSTANT  # decimal, not a hex type byte


def test_metadata_facts_from_the_metadata_chunk():
    chunks = chunk_document("eip-905", make_doc(905))
    meta = {f.name: f for f in extract_facts(chunks) if f.kind == METADATA}
    assert meta["created"].value == "2021-07-17" or meta["created"].value.startswith("2021-")
    assert meta["requires"].tokens == ("903", "904")
    assert meta["requires"].chunk_id == "eip-905#000"


def test_metadata_without_requires_has_no_requires_fact():
    chunk = Chunk("eip-9#000", "eip-9", 9, "T", "Metadata", "prose",
                  "EIP-9: T / Metadata\neip: 9\ntitle: T\nstatus: Draft\ncreated: 2020-01-02")
    names = [f.name for f in extract_facts([chunk])]
    assert names == ["created"]


def test_plain_strips_markup_but_not_exponents():
    assert plain("A `code` and **bold** word") == "A code and bold word"
    assert plain("cost 2**17 and 3**4 gas") == "cost 2**17 and 3**4 gas"


def test_paragraph_sentences_rejoin_hard_wrapped_lines_but_split_bullets():
    body = "This sentence is wrapped\nacross two lines. Next one.\n\n- item one\n- item two"
    assert paragraph_sentences(body) == [
        "This sentence is wrapped across two lines.", "Next one.", "item one", "item two"]


def test_cloze_blanks_exactly_one_number_per_candidate():
    chunk = mk("The widget limit is set to 4096 units for every single account on the network.")
    cands = cloze_candidates(chunk)
    assert len(cands) == 1
    sentence, blanked, value = cands[0]
    assert value == "4096" and blanked.count(BLANK) == 1 and "4096" not in blanked
    assert blanked.replace(BLANK, "4096") == sentence


@pytest.mark.parametrize("text", [
    "See EIP-2718 for the envelope format used by every transaction in this document.",
    "The network upgrade happened in 2021 after a long period of testing and review.",
    "Short 5 sentence.",
    "A table row | with 4096 | pipes in it that should never be used as a cloze source ok.",
    "This sentence mentions https://example.org/4096 and is long enough to pass the length rule.",
    "Values such as 0xcccc and similar placeholders are not real constants in this sentence.",
    "The list goes one, two, three... and then continues with 4096 further items after that.",
    "lowercase start sentences are not used as cloze sources even with 4096 in them here.",
    "The cap is exactly 2^24 and nothing else is said about the value of it in this sentence.",
    "The expected time for an attacker holding a third of the stake is 205 000 years in total.",
    "Using the 900 standard you could mint millions of blank tokens in a cost effective way here.",
    "Created on 2021-04-13 by somebody who wrote this particular sentence for the test suite.",
    "Version 3 of RFC 2119 and BIP 32 and ISO 8601 are cited by this fairly long sentence here.",
])
def test_cloze_rejects_unsuitable_sentences(text):
    assert cloze_candidates(mk(text)) == []


def test_cloze_skips_code_and_metadata_chunks():
    assert cloze_candidates(mk("The widget limit is set to 4096 units for every account today.",
                               kind="code")) == []
    assert cloze_candidates(mk("The widget limit is set to 4096 units for every account today.",
                               section="Metadata")) == []


def test_duplicate_blanked_sentences_in_a_chunk_are_dropped():
    s = "The widget limit is set to 4096 units for every single account on the network."
    facts = [f for f in extract_facts([mk(f"{s}\n\n{s}")]) if f.kind == CLOZE]
    assert facts == []


def test_supports_named_requires_name_and_value():
    chunk = mk("| `MAX_THING` | `64` |")
    fact = Fact("f", "eip-900", chunk.id, CONSTANT, "MAX_THING", "64", ("64",))
    assert supports(chunk, fact)
    assert not supports(mk("| `OTHER_THING` | `64` |"), fact)  # right value, wrong name
    assert not supports(mk("| `MAX_THING` | `65` |"), fact)  # right name, wrong value


def test_supports_accepts_equivalent_spellings():
    chunk = mk("| `MAX_THING` | `0x40` |")
    fact = Fact("f", "eip-900", chunk.id, CONSTANT, "MAX_THING", "64", ("64",))
    assert supports(chunk, fact)


def test_supports_metadata_requires_matching_doc_and_section():
    meta = Chunk("eip-9#000", "eip-9", 9, "T", "Metadata", "prose",
                 "EIP-9: T / Metadata\nrequires: 5, 6")
    fact = Fact("f", "eip-9", meta.id, METADATA, "requires", "5, 6", ("5", "6"))
    assert supports(meta, fact)
    other_doc = Chunk("eip-8#000", "eip-8", 8, "T", "Metadata", "prose",
                      "EIP-8: T / Metadata\nrequires: 5, 6")
    assert not supports(other_doc, fact)
    assert not supports(mk("requires: 5, 6"), fact)


def test_supports_cloze_needs_sentence_overlap_and_value():
    s = "The widget limit is set to 4096 units for every single account on the network."
    chunk = mk(s)
    [fact] = [f for f in extract_facts([chunk]) if f.kind == CLOZE]
    assert supports(chunk, fact)
    assert not supports(mk("Completely different words appear here with 4096 only."), fact)
    assert not supports(mk(s.replace("4096", "8192")), fact)


def test_token_supported_uses_variants_only_for_numbers():
    assert token_supported("0x40", {"64"}) and token_supported("64", {"0x40"})
    assert not token_supported("max_thing", {"other"})
    assert token_supported("max_thing", {"max_thing"})


def test_fact_dict_roundtrip_and_doc_number():
    f = Fact("a:b:c", "eip-1559", "eip-1559#001", CONSTANT, "N", "1", ("1",))
    assert Fact.from_dict(f.to_dict()) == f and f.doc_number == 1559


def test_extract_facts_is_sorted_and_unique_on_synthetic_corpus():
    facts = extract_facts(make_chunks(range(900, 906)))
    ids = [f.id for f in facts]
    assert ids == sorted(set(ids), key=lambda i: ids.index(i))
    assert Counter(f.kind for f in facts)[CONSTANT] >= 6
    keys = [(f.doc_number, f.chunk_id, f.id) for f in facts]
    assert keys == sorted(keys)


def test_every_extracted_fact_is_supported_by_its_own_chunk(dataset):
    bad = [f.id for f in dataset.facts
           if not supports(dataset.index.by_id[f.chunk_id], f, dataset.index.tokens(f.chunk_id))]
    assert bad == []


def test_real_corpus_yields_all_fact_kinds_in_volume(dataset):
    counts = Counter(f.kind for f in dataset.facts)
    assert counts[CLOZE] > 500 and counts[METADATA] > 100
    assert counts[CONSTANT] > 80 and counts[OPCODE] > 20 and counts[TYPE_BYTE] >= 3
    assert {f.doc for f in dataset.facts} <= {c.doc for c in dataset.index.chunks}


def test_real_fact_tokens_occur_in_their_chunk(dataset):
    for f in dataset.facts:
        toks = dataset.index.tokens(f.chunk_id)
        assert all(token_supported(t, toks) for t in f.tokens), f.id
        assert f.tokens == tuple(sorted(f.tokens)) and f.tokens
        assert claim_tokens(f.value) >= set(f.tokens) or f.kind == METADATA
