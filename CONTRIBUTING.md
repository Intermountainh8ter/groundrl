# Contributing

Thanks for looking. This is a research repository: the most useful contributions are a new
reward-hacking attack, a flaw in the verifier, or a real training result.

## Ground rules

* **No fabricated or simulated results.** `results/*.json` may only be produced by the commands
  that name them (`reward_audit.py`, `groundrl baseline-eval`, `train/eval.py`). Output of a
  `--mock` run or of tests is labelled MOCK and is never a result. Do not edit a results file by
  hand. `scripts/make_report.py` renders only what is on disk.
* **The core library stays on the standard library plus numpy.** torch, transformers, trl, peft,
  datasets and vllm are optional extras, imported lazily, and tests must not require them.
* Python 3.10 or newer syntax only. CI runs 3.10 to 3.13.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
make test lint
```

`make cov` runs the suite with coverage; the project keeps at least 90% line coverage on `src/`.

## Adding an attack to the audit

An attack is a function `(example, rng) -> answer string` in `src/groundrl/audit.py`, registered in
`POLICIES` with a category (`hack` or `extra`), a one-line description and whether it uses gold.
Then:

1. Add a test in `tests/test_audit.py` showing the policy does what its description says.
2. Run `python scripts/reward_audit.py --split val` first. Do not tune the reward on TEST.
3. If it is exploitable under v2 (mean at least 0.50), that is a finding. Report it; if you also
   fix it, add an entry to the revision log in `docs/METHODOLOGY.md` with the before and after
   numbers, and keep the old behaviour reproducible through a `V2Config` preset.
4. Regenerate the committed audit (`python scripts/reward_audit.py --split test` and `--split val`)
   and `groundrl baseline-eval --split test`, then `python scripts/make_report.py`. Update the
   tables in `README.md`; `tests/test_docs_numbers.py` fails if the documentation drifts from the
   result files.

## Changing the data pipeline

The task files are deterministic: `groundrl build-tasks` must reproduce `data/` byte for byte, and a
test compares a fresh generation with the committed files. If you change chunking, fact extraction
or generation, rebuild, rerun the audit and baselines, and say so in `CHANGELOG.md`.

## Style

`ruff check .` must pass (line length 100). Prefer small pure functions; every function that
touches rewards or verification gets a test for its edge cases. No placeholders or TODOs in code.

## Reporting a verifier flaw

Open an issue with the answer string, the context chunks and what you expected. Flaws that make a
wrong answer verify are welcome; they belong in `tests/test_verifier.py` either as fixed behaviour
or as a documented blind spot.
