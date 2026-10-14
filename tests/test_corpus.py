from __future__ import annotations

import json

import pytest

from groundrl import corpus
from groundrl.corpus import (
    CURATED_EIPS,
    MIN_BYTES,
    fetch_corpus,
    fetch_one,
    front_matter,
    load_documents,
    sha256_hex,
    urllib_opener,
    verify_manifest,
)

BIG = b"x" * 400
STUB = b"---\neip: 20\nstatus: Moved\n---\nmoved"


def fake_opener(mapping):
    def opener(url):
        if url not in mapping:
            raise FileNotFoundError(url)
        return mapping[url]

    return opener


def test_curated_list_is_unique_and_reasonably_sized():
    assert len(set(CURATED_EIPS)) == len(CURATED_EIPS)
    assert 100 <= len(CURATED_EIPS) <= 160
    assert all(isinstance(n, int) and n > 0 for n in CURATED_EIPS)


def test_fetch_one_prefers_eips_repository():
    url = corpus.EIPS_URL.format(n=1559)
    res = fetch_one(1559, fake_opener({url: BIG}))
    assert res.ok and res.source == url and res.content == BIG


def test_fetch_one_falls_back_to_ercs_for_moved_stub():
    eips = corpus.EIPS_URL.format(n=20)
    ercs = corpus.ERCS_URL.format(n=20)
    res = fetch_one(20, fake_opener({eips: STUB, ercs: BIG}))
    assert res.ok and res.source == ercs


def test_fetch_one_skips_when_both_missing_or_small():
    res = fetch_one(5, fake_opener({}))
    assert not res.ok and "404" in res.reason
    eips = corpus.EIPS_URL.format(n=6)
    res = fetch_one(6, fake_opener({eips: STUB}))
    assert not res.ok and "bytes" in res.reason


def test_fetch_corpus_writes_files_and_manifest(tmp_path):
    mapping = {corpus.EIPS_URL.format(n=1): BIG, corpus.EIPS_URL.format(n=2): STUB}
    manifest = fetch_corpus([2, 1, 1, 3], tmp_path / "raw", fake_opener(mapping))
    assert [d["number"] for d in manifest["documents"]] == [1]
    assert {s["number"] for s in manifest["skipped"]} == {2, 3}
    on_disk = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert on_disk["documents"][0]["sha256"] == sha256_hex(BIG)
    assert verify_manifest(tmp_path) == []


def test_verify_manifest_detects_tampering_missing_and_unlisted(tmp_path):
    mapping = {corpus.EIPS_URL.format(n=1): BIG, corpus.EIPS_URL.format(n=7): BIG + b"y"}
    fetch_corpus([1, 7], tmp_path / "raw", fake_opener(mapping))
    (tmp_path / "raw" / "eip-1.md").write_bytes(b"z" * 400)
    (tmp_path / "raw" / "eip-7.md").unlink()
    (tmp_path / "raw" / "eip-99.md").write_bytes(BIG)
    problems = " ".join(verify_manifest(tmp_path))
    assert "sha256 mismatch for eip-1.md" in problems
    assert "missing eip-7.md" in problems
    assert "unlisted file eip-99.md" in problems


def test_verify_manifest_detects_size_mismatch_and_small_files(tmp_path):
    fetch_corpus([1], tmp_path / "raw", fake_opener({corpus.EIPS_URL.format(n=1): BIG}))
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    manifest["documents"][0]["bytes"] = 5
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert any("size mismatch" in p for p in verify_manifest(tmp_path))
    small = b"s" * (MIN_BYTES - 1)
    (tmp_path / "raw" / "eip-1.md").write_bytes(small)
    manifest["documents"][0].update(bytes=len(small), sha256=sha256_hex(small))
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert any("below the minimum" in p for p in verify_manifest(tmp_path))


def test_urllib_opener_refuses_other_hosts():
    opener = urllib_opener()
    with pytest.raises(ValueError, match="refusing"):
        opener("https://huggingface.co/some/file")


def test_front_matter_parsing():
    meta = front_matter("---\neip: 1559\ntitle: Fee market\nrequires: 2718, 2930\n---\nbody")
    assert meta == {"eip": "1559", "title": "Fee market", "requires": "2718, 2930"}
    assert front_matter("no header here") == {}


def test_load_documents_sorts_numerically(tmp_path):
    for n in (100, 20, 3):
        (tmp_path / f"eip-{n}.md").write_text(f"doc {n}")
    assert list(load_documents(tmp_path)) == ["eip-3", "eip-20", "eip-100"]


def test_real_corpus_matches_manifest(data_dir):
    assert verify_manifest(data_dir) == []
    manifest = corpus.read_manifest(data_dir / "MANIFEST.json")
    assert len(manifest["documents"]) >= 100
    assert all(d["bytes"] > MIN_BYTES for d in manifest["documents"])
    assert all(d["source"].startswith("https://raw.githubusercontent.com/")
               for d in manifest["documents"])
