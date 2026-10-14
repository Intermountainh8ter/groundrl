from __future__ import annotations

import pytest

from groundrl.chunker import (
    Chunk,
    chunk_corpus,
    chunk_document,
    clean_markdown,
    split_blocks,
    split_code,
    split_prose,
    split_sections,
    split_table,
)
from groundrl.corpus import load_documents
from tests.helpers import make_doc


def test_chunk_ids_are_stable_zero_padded_and_unique():
    chunks = chunk_document("eip-905", make_doc(905))
    ids = [c.id for c in chunks]
    assert ids[0] == "eip-905#000" and ids[1] == "eip-905#001"
    assert len(set(ids)) == len(ids)
    assert chunk_document("eip-905", make_doc(905)) == chunks


def test_metadata_chunk_carries_checkable_fields():
    meta = chunk_document("eip-905", make_doc(905))[0]
    assert meta.section == "Metadata"
    assert "requires: 904, 903" in meta.text and "status: Final" in meta.text
    assert meta.text.splitlines()[0] == "EIP-905: Synthetic proposal 905 / Metadata"


def test_every_chunk_starts_with_a_context_header():
    for c in chunk_document("eip-905", make_doc(905)):
        assert c.text.startswith("EIP-905: Synthetic proposal 905 / ")
        assert c.body and c.body in c.text


def test_copyright_section_is_dropped():
    chunks = chunk_document("eip-905", make_doc(905))
    assert not any(c.section.lower() == "copyright" for c in chunks)
    assert not any("waived via CC0" in c.text for c in chunks)


def test_headings_inside_code_fences_are_not_sections():
    body = "## A\ntext\n```\n# not a heading\n```\n## B\nmore"
    assert [h for h, _ in split_sections(body)] == ["A", "B"]


def test_table_is_atomic_when_small_and_split_with_header_when_large():
    rows = "\n".join(f"| `K{i}` | `{i}` |" for i in range(40))
    table = f"| Name | Value |\n| - | - |\n{rows}"
    assert split_table(table, 500) == [table]
    pieces = split_table(table, 30)
    assert len(pieces) > 1
    for p in pieces:
        assert p.startswith("| Name | Value |\n| - | - |")
    joined = "\n".join(line for p in pieces for line in p.splitlines()[2:])
    assert joined == rows


def test_code_block_is_windowed_by_lines_and_keeps_fences():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(60)) + "\n```"
    pieces = split_code(code, 30)
    assert len(pieces) > 1
    assert all(p.startswith("```python") and p.endswith("```") for p in pieces)
    assert split_code("```\nshort\n```", 30) == ["```\nshort\n```"]


def test_giant_code_line_is_hard_wrapped():
    code = "```\n" + " ".join(["0xab"] * 200) + "\n```"
    pieces = split_code(code, 50)
    assert len(pieces) >= 4 and all(len(p.split()) <= 52 for p in pieces)


def test_prose_splits_on_sentences_not_mid_sentence():
    text = " ".join(f"Sentence number {i} has the value {i * 11}." for i in range(30))
    pieces = split_prose(text, 40)
    assert len(pieces) > 1
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


def test_one_oversized_sentence_is_hard_wrapped():
    sentence = " ".join(["word"] * 100) + "."
    pieces = split_prose(f"Short one. {sentence}", 30)
    assert max(len(p.split()) for p in pieces) <= 30


def test_split_blocks_recognises_prose_code_and_tables():
    text = "Intro line.\n\n```\ncode\n```\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nTail."
    kinds = [k for k, _ in split_blocks(text)]
    assert kinds == ["prose", "code", "table", "prose"]


def test_unterminated_fence_does_not_crash():
    blocks = split_blocks("```\nnever closed\nstill code")
    assert blocks[0][0] == "code"


def test_clean_markdown_rewrites_links_to_proposals():
    text = "See [2718](./eip-2718.md) and [ERC-20][erc-20] and [site](https://x.org) <!-- c -->."
    out = clean_markdown(text)
    assert "EIP-2718" in out and "EIP-20" in out and "site" in out
    assert "https" not in out and "<!--" not in out


def test_chunk_size_bound_and_kinds_on_synthetic_doc():
    chunks = chunk_document("eip-905", make_doc(905), max_words=40)
    assert max(len(c.body.split()) for c in chunks) <= 45
    assert any("| - | - |" in c.text for c in chunks)
    assert any(c.text.count("```") == 2 for c in chunks)  # fences stay paired in one chunk


def test_invalid_document_id_rejected():
    with pytest.raises(ValueError, match="document id"):
        chunk_document("rfc-1", "---\n---\nbody")


def test_chunk_dict_roundtrip():
    c = chunk_document("eip-905", make_doc(905))[2]
    assert Chunk.from_dict(c.to_dict()) == c


def test_chunk_corpus_concatenates_documents():
    docs = {"eip-905": make_doc(905), "eip-906": make_doc(906)}
    ids = [c.id for c in chunk_corpus(docs)]
    assert ids[0].startswith("eip-905#") and ids[-1].startswith("eip-906#")


def test_committed_chunks_match_a_fresh_chunking(data_dir, dataset):
    docs = load_documents(data_dir / "raw")
    fresh = chunk_corpus(docs)
    assert [c.to_dict() for c in fresh] == [c.to_dict() for c in dataset.index.chunks]


def test_real_chunks_have_no_oversized_or_empty_chunks(dataset):
    sizes = [len(c.body.split()) for c in dataset.index.chunks]
    assert min(sizes) >= 1 and max(sizes) <= 140
