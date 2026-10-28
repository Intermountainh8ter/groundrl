# Changelog

All notable changes. Dates are ISO. Version 0.1.0 is the first and only release so far.

## Unreleased

### Changed

* Launch copy (`docs/launch/`) rewritten around the real results: one SFT run (Qwen3-0.6B, LoRA,
  AMD gfx906) and one degenerate GRPO run reported as a null result, with the limits stated next
  to the numbers. The launch-copy tests now derive the quoted numbers from `results/`.
* README, threats to validity, runbook and flowchart text no longer say that nothing has been
  trained; `results/REPORT.md` lists the GRPO run on record.

## 0.1.0 - 2026-10-14

First public state of the repository. **No model has been trained.**

### Added

* Corpus: curated list of 130 EIP numbers (all resolved), downloader restricted to `raw.githubusercontent.com`
  with an ERCs-repository fallback for moved documents, sha256 manifest; 130 documents kept.
* Section/table/code-aware chunker with stable ids (`eip-N#CCC`) and a metadata chunk per document.
* Deterministic fact extraction (constants, opcodes, type bytes, preamble fields, blanked
  sentences) and a seeded task generator: five question types, split strictly by EIP, invariants
  validated (answerable tasks have a supporting chunk, unanswerable ones have none, the gold
  answer passes the verifier and never trips the consistency term).
* Verifier with per-claim provenance; rejects empty answers, spoofed or out-of-context citations,
  unsupported claim tokens, and invisible, look-alike or non-ASCII digit/letter characters.
* Rewards v0 (binary), v1 (graded) and v2 (graded plus relevance, correctness, citation quality,
  length, redundancy, consistency and abstention calibration) as pure functions, with TRL-style
  wrappers and training diagnostics.
* Reward-hacking audit: 19 scripted policies scored under v0, v1, v2, one-term-removed ablations,
  the first v2 design (`rev1`) and a gold-free v2. Committed for TEST and VAL.
* Scripted baselines (`groundrl baseline-eval`) through the same evaluation pipeline as a model.
* Training pipeline, never run on real hardware: SFT (LoRA, merge), GRPO, evaluation with bootstrap
  intervals, four JSON configs (Qwen3-0.6B and 1.7B at 24 GB and 12 GB), `--dry-run` without torch,
  `--mock` plumbing runs labelled MOCK, report renderer, Windows and Linux runbook.
* Documentation: methodology with the reward revision log, threats to validity, flowcharts
  (usage and troubleshooting), launch copy. CI for Python 3.10 to 3.13, Makefile, Dockerfile.

### Changed (during development, recorded for honesty)

* Reward v2 revision 2: the first design was exploitable by two hedging policies in the audit
  (0.63); correctness now costs a factor of 0.25 per extra claim and a consistency term was added.
  Details and the rejected intermediate attempt are in `docs/METHODOLOGY.md`.
* Task generation: cloze extraction fixed for numbers split by a space and for a document naming
  itself by number; tasks whose gold answer would count as hedging are not generated.
