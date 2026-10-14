# data

* `raw/eip-N.md` - proposals fetched from `raw.githubusercontent.com` (CC0-1.0). Files that were
  moved out of the EIPs repository (the ERC category) are redirect stubs there, so for those the
  ERCs repository is used; `MANIFEST.json` records the exact source URL and sha256 of every file.
  Re-fetch with `python scripts/fetch_corpus.py`.
* `chunks.jsonl`, `facts.jsonl`, `tasks/*.jsonl` - derived and deterministic. Rebuild with
  `groundrl build-tasks`; `groundrl check-data` verifies hashes and task invariants.
* `tasks/SUMMARY.json` - counts per split and question type, the seed, and the sha256 of each task file.

Splits are made by EIP: no EIP appears in more than one of train, val and test, as a gold source,
as a comparison partner or as a distractor chunk.
