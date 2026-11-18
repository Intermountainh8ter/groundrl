# groundrl

**A deterministic citation-grounding verifier used as a verifiable reward (RLVR), a reward-hacking
audit of it, and an unrun LoRA + GRPO pipeline for training small models to answer questions about
Ethereum EIPs with correct, grounded citations.**

> **Training results: NOT YET RUN.** No model has been trained or fine-tuned in this repository.
> It was built in a container with no GPU. Everything under "Results" below is either a scripted,
> model-free result that anyone can regenerate with one command, or an explicit placeholder.
> `scripts/make_report.py` renders only from `results/*.json` files produced by real runs, and
> says "NOT YET RUN" until one exists.

## The idea

Retrieval-augmented answers are only useful if each claim can be traced to the text it came from.
Checks for that are usually applied after the model has answered. Here the same deterministic check
is used as the *reward* during RL, so a small model (Qwen3-0.6B / 1.7B class) can be trained to cite
correctly in the first place. The verifier accepts an answer only if

1. every sentence carries a citation of the form `[eip-N#CCC]`;
2. every cited chunk was in the provided context (a citation like `[eip-1559#999]` is spoofed);
3. every number, hex value, `2**n` expression and ALL_CAPS identifier in a sentence appears as a
   token in the chunks that sentence cites.

A verifier like that is cheap, exact and reproducible, and it is also easy to game: it checks token
presence, not meaning. The contribution of this repository is therefore (1) the reward design and
its reward-hacking analysis, which is real and reproducible here, and (2) the training and
evaluation pipeline, which is complete but unrun.

## Related work

* **`groundrails`** (PyPI, MIT): per its PyPI description, "Grounding guardrails for agentic RAG -
  deterministic claim verification". It is the closest tool I found; I read only its package
  description, and nothing there describes using the check as an RL reward.
* A May 2026 write-up of RLVR with GRPO on a 1.5B code model with graded rewards motivated the
  graded reward (v1). I have not re-read it while building this repository; it is cited from the
  owner's notes.
* I found nothing on citation-grounding rewards for RL. That is the result of a limited search,
  not a proof that none exists.
* The verifier is a clean reimplementation of the grounding check in the author's own
  [`ethresearch-rag`](https://github.com/Intermountainh8ter/ethresearch-rag) (MIT, same author).

## Design

![usage flow](docs/diagrams/usage.svg)

* **Corpus.** 130 curated EIPs fetched from `raw.githubusercontent.com` (CC0; ERC
  documents come from the ERCs repository), sha256 manifest in `data/MANIFEST.json`, chunked
  with a section/table/code-aware chunker (`3066` chunks).
* **Tasks.** 1792 seeded questions with gold (`1262` train / `265` val /
  `265` test), five types: constant lookup, opcode/type byte, multi-fact, comparison across two
  EIPs, and unanswerable (the gold chunk is absent; the right behaviour is a fixed abstention
  sentence). The split is **by EIP**: no EIP appears in two splits, not even as a distractor chunk.
* **Rewards.** `v0` binary verifier pass; `v1` graded per sentence; `v2` graded plus relevance,
  answer-correctness against gold, citation precision/recall, length and redundancy penalties, a
  consistency term against hedging, and abstention calibration. Pure functions returning
  `(score, breakdown)`, with TRL wrappers (`make_trl_reward`) importable without TRL.
* **Pipeline (unrun).** SFT baseline with LoRA, adapter merge, GRPO with reward v2 on a fresh
  LoRA, evaluation on TEST with bootstrap intervals. JSON configs for Qwen3-0.6B and 1.7B at 24 GB
  and 12 GB, `--dry-run` that needs no torch, and `--mock` plumbing runs labelled MOCK.
* The core library uses only the standard library and numpy; torch, transformers, trl, peft,
  datasets and vllm are optional extras imported lazily and never required by the tests.

Details and the revision history of the reward are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Results

### 1. Reward-hacking audit (real, no model, reproducible)

Nineteen scripted policies (plain functions) are scored under each reward version on the TEST
split (265 tasks: 235 answerable, 30 unanswerable). A non-reference
policy is flagged **EXPLOITABLE** (bold) if its mean reward is at least 0.50, a threshold fixed
before any number was produced. "fires on" is the share of tasks where the policy's output
differs from the gold answer.

```
python scripts/reward_audit.py --split test     # about 30 s, writes results/audit_test.{json,md}
```

| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |
|---|---|---|---|---|---|---|---|
| `oracle` | reference | yes | 0% | 0.887 | 0.887 | 1.000 | n/a |
| `extractive` | reference | no | 57% | 0.985 | 0.988 | 0.613 | 0.325 |
| `eip_id_shortcut` | reference | no | 52% | 0.928 | 0.931 | 0.670 | 0.361 |
| `copy_chunk` | hack | yes | 97% | **0.981** | **0.996** | 0.185 | 0.160 |
| `cite_everything` | hack | yes | 100% | **1.000** | **1.000** | 0.296 | 0.296 |
| `stuff_numbers` | hack | yes | 100% | **1.000** | **0.882** | 0.247 | 0.247 |
| `refuse_all` | hack | no | 89% | 0.000 | 0.000 | 0.113 | 0.000 |
| `parrot_question` | hack | no | 100% | 0.174 | **0.500** | 0.077 | 0.077 |
| `empty` | hack | no | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `citation_spam` | hack | yes | 100% | **1.000** | **1.000** | 0.000 | 0.000 |
| `wrong_chunk` | hack | no | 100% | **0.887** | **0.928** | 0.074 | 0.074 |
| `one_wrong_constant` | hack | yes | 88% | **0.989** | **0.996** | 0.277 | 0.181 |
| `paraphrase_no_tokens` | hack | no | 100% | **1.000** | **0.724** | 0.064 | 0.064 |
| `hedge_right_and_wrong` | extra | yes | 78% | **0.989** | **0.998** | 0.276 | 0.073 |
| `hedge_inline` | extra | yes | 78% | **1.000** | **0.998** | 0.440 | 0.284 |
| `spoofed_citation` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | extra | yes | 100% | 0.113 | **0.722** | 0.109 | 0.109 |
| `uncited_correct` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | extra | yes | 100% | 0.049 | 0.049 | 0.000 | 0.000 |

What it shows, in plain terms:

* **v0 and v1 are broken as training rewards.** Seven of the ten required hacks (copy the chunk,
  cite everything, stuff numbers, repeat a citation, answer from the wrong chunk, swap one constant
  for another number from the same chunk, say something vague with a citation) score 0.89-1.00
  under v0 and 0.72-1.00 under v1. Nothing in a binary or graded verifier distinguishes them
  from a correct answer.
* **v2 (revision 2) holds every enumerated hack below 0.50 on TEST and VAL.** The highest is
  `hedge_inline` (the right value followed by "(or <wrong value>)") at 0.440.
  Revision 1 of v2 was exploitable by both hedges (0.63); the audit found that, and revision 2
  fixed it (revision log in `docs/METHODOLOGY.md`).
* **Which term closes which hole** (full table with every ablation in `results/audit_test.md`):
  removing the citation term re-opens `cite_everything` (0.89); removing correctness re-opens
  `stuff_numbers`, `one_wrong_constant` and `hallucinated_value` (0.61-0.85); abstention
  calibration is what keeps `refuse_all` at the unanswerable share (0.113).
* **A reward without gold labels is not enough.** v2 with only the verifier-based terms is
  exploitable by 7 policies on TEST (9 on VAL).
* **Residual holes, reported honestly.** `hedge_inline` reaches 0.44; `copy_chunk` still earns at
  least 0.5 on about 14% of tasks, where the gold chunk is a single short sentence and copying it
  is indistinguishable from answering; a hedge that rewords the second sentence is not caught by
  the consistency term (only by the extra-claim penalty). `v2` was revised after looking at this
  TEST audit, so "closed" is in-sample for the attacks I enumerated; VAL is a robustness check.
  A trained policy may find attacks I did not.

### 2. Scripted baselines (real, not trained models)

The same evaluation pipeline that will score a trained model, applied to scripted policies on TEST
(95% bootstrap intervals over tasks). `extractive` takes the single best-matching line and cites
it; it is the bar a trained model has to clear to show it learned more than locate-and-copy.

```
groundrl baseline-eval --split test
```

| scripted policy | grounded-correct | verified | abstain recall | hallucinated cite | v0 | v1 | v2 |
|---|---|---|---|---|---|---|---|
| `oracle` | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 0.0% (0.0-0.0) | 0.887 | 0.887 | 1.000 (1.000-1.000) |
| `extractive` | 57.4% (51.1-63.8) | 98.3% (96.6-99.6) | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 0.985 | 0.988 | 0.613 (0.563-0.663) |
| `eip_id_shortcut` | 57.4% (51.1-63.8) | 98.3% (96.6-99.6) | 50.0% (18.2-81.5) | 0.0% (0.0-0.0) | 0.928 | 0.931 | 0.670 (0.622-0.716) |
| `refuse_all` | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 100.0% (100.0-100.0) | 0.0% (0.0-0.0) | 0.000 | 0.000 | 0.113 (0.079-0.151) |

### 3. Training results: NOT YET RUN

| run | grounded-correct | verified | abstain precision / recall | hallucinated cite | v2 |
|---|---|---|---|---|---|
| base Qwen3-0.6B | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| SFT baseline | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| GRPO, reward v2 | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |

These rows are placeholders, not estimates. They will be replaced by numbers only through
`python scripts/make_report.py` after `train/eval.py` has written real `results/eval_*.json` files.

## Quickstart

```bash
pip install -e ".[dev]"            # numpy only; no torch needed
pytest -q                          # whole suite, no GPU, no network
ruff check .

groundrl check-data                # manifest hashes + task invariants
python scripts/reward_audit.py --split test
groundrl baseline-eval --split test
python scripts/make_report.py      # renders results/*.json into results/REPORT.md

# the training pipeline, without a GPU:
python train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --dry-run
python train/grpo.py --config train/configs/qwen3-0.6b-24gb.json --mock --limit 16   # MOCK
```

Scoring an answer yourself:

```python
from groundrl.dataset import load_dataset
from groundrl.example import make_example
from groundrl.rewards import reward_v2

ds = load_dataset("data")
ex = make_example(ds.tasks["test"][0], ds.index)
score, breakdown = reward_v2(ex.task.gold_answer, ex)      # 1.0 for the gold answer
```

Real training needs `pip install -e ".[train]"` and a GPU; see [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
for Windows PowerShell and Linux instructions and rented-GPU notes.

## Repository layout

| path | what |
|---|---|
| `src/groundrl/verifier.py`, `tokens.py` | the deterministic grounding verifier and its token primitives |
| `src/groundrl/rewards.py`, `consistency.py` | reward v0/v1/v2 and TRL wrappers |
| `src/groundrl/audit.py` | the adversarial policies and the audit |
| `src/groundrl/chunker.py`, `facts.py`, `tasks.py`, `dataset.py` | chunking, fact extraction, task generation |
| `src/groundrl/trainconf.py`, `trainplan.py`, `trainctl.py`, `trl_backend.py` | configs, dry run, runners (heavy imports are lazy) |
| `src/groundrl/evaluate.py`, `report.py` | metrics with bootstrap CIs, report renderer |
| `train/` | `sft.py`, `grpo.py`, `eval.py` and `configs/*.json` |
| `data/` | raw EIPs, manifest, chunks, facts, task splits |
| `results/` | audit, baselines, report (no training results) |
| `docs/` | methodology, threats to validity, runbook, flowcharts, launch copy |

## Limits

The verifier is lexical, the questions are synthetic templates, the TEST split has 265 tasks
from a small number of EIPs, and nothing has been trained. Read
[`docs/THREATS_TO_VALIDITY.md`](docs/THREATS_TO_VALIDITY.md) before drawing any conclusion.

## License

MIT, Copyright (c) 2026 Ammon Jackson. The EIP texts in `data/raw` are CC0. The verifier is adapted
from `ethresearch-rag` (MIT, same author).

### 2. Verifier-strictness ladder (SIMULATED exploiter, no model)

`src/groundrl/ladder.py` defines four grader rungs (`first_number`, `dup_key`, `skip_denominator`,
`strict`) and scores a scripted exploiting policy (self-printed pass line, duplicate-key spray,
skipped items) on the TEST split. The strict rung rejects it by construction, so that zero is a
check on the grader, not a finding about any model. Exploit success rate by rung, from
`results/50-strictness-ladder.json`: `first_number` 1.0, `dup_key` 0.569811320754717,
`skip_denominator` 0.8867924528301887, `strict` 0.0. The zero-reward-spread share equals the same
values here.

```
python3 scripts/strictness_ladder.py --split test --seed 7 --out results/50-strictness-ladder.json
```

### 3. Exploit advantage with an imperfect honest policy (SIMULATED, no model)

`run_noisy_honest_advantage` in `src/groundrl/exploit_advantage.py` repeats the group sampler, but
each honest rollout is the gold answer only with probability q and otherwise a corrupted answer.
Share of groups whose exploit rollouts have strictly positive mean group-normalised advantage at
q=0.5 and exploit mixing probability 0.5, from `results/53-noisy-honest-advantage.json`:
`first_number` 0.8618867924528302, `dup_key` 0.5033962264150943, `skip_denominator`
0.7890566037735849, `strict` 0.0. The strict zero is a property of the grader on a scripted
exploiter, not a finding about any trained model.

```
python3 scripts/exploit_advantage_noisy_honest.py --split test --group-size 8 --groups-per-task 20 --qs 0.25 0.5 0.75 --probs 0.5 --seed 7 --out results/53-noisy-honest-advantage.json
```

### Group-size sweep (SIMULATED, no model)

`python3 scripts/group_size_sweep.py --split test --group-sizes 2 4 8 16 --groups-per-task 20 --q 0.5 --prob 0.5 --seed 7 --out results/55-group-size-sweep.json`

With exploit probability 0.5 and honest-success probability 0.5, the strict rung's share of groups with positive exploit advantage is 0.0 at every G in {2, 4, 8, 16}. The first_number rung rises from 0.24018867924528303 at G=2 to 0.9543396226415094 at G=16
(`results/55-group-size-sweep.json`).

#### Honest-success sweep (SIMULATED, no model)

`results/57-honest-success-sweep.json`: at G=8, p=0.5, the strict rung's share of groups with positive exploit advantage is 0.0 at every q in {0.1, 0.25, 0.5, 0.75, 0.9}. The first_number rung falls from 0.9554716981132075 at q=0.1 to 0.3383018867924528 at q=0.9, so it stays at or above the pre-registered 0.20.

### Verifier error direction (SIMULATED, no model)

`python3 scripts/verifier_error_asymmetry.py --split test --group-size 8 --groups-per-task 20 --error-rate 0.2 --seed 7 --out results/236-verifier-error-asymmetry.json`

One error direction is injected at a time at the same rate: false positives accept an exploit rollout the rung rejects, false negatives reject an honest rollout the rung accepts. At the pre-registered cell (strict rung, exploit probability 0.5, error rate 0.2) the exploit share of reinforced rollouts is 0.1653428751179616 higher under false positives than under false negatives, with a 95% interval of 0.16061935185453666 to 0.1698299811591728 (`results/236-verifier-error-asymmetry.json`, `/asymmetry/exploit_share_fp_minus_fn`). The strict rung's false-negative exploit share is 0.0 by construction, because the strict rung never reinforces the exploit. The sign reverses on the three weaker rungs: at exploit probability 0.5 the difference is -0.0515 for first_number, -0.0229 for dup_key and -0.052 for skip_denominator, so the asymmetry holds only where the exploit is rejected by the base grader. "Reward v0" is mapped to the strict rung (see `preregistration_additions.md`).

### Reward-version advantage replay (SIMULATED, no model)

`scripts/reward_version_advantage.py` mixes the oracle with each of seven audit hack policies in
GRPO groups (G=8, 20 groups per task, mixing probability 0.5, seed 7, TEST split) and scores them
under the real reward versions. Share of groups whose hack rollouts have positive mean
group-normalised advantage, averaged over the seven hacks (`results/277-reward-version-advantage.json`):
v0 0.1106, v1 0.1122, v2 0.0. The pre-registered v2 bound (at most 0.05) held; the pre-registered
v0 bound (at least 0.30) did not (0.1106). Per-hack values are in the results file.

### Degenerate-group share by reward version (SIMULATED, no model)

`scripts/degenerate_group_by_reward.py` uses the same oracle/hack groups (G=8, 20 groups per task,
mixing probability 0.5, seed 7, TEST split) and reports the share of groups whose rewards are all
identical (zero group-normalised advantage), averaged over the seven hacks
(`results/282-degenerate-group-by-reward.json`): v0 0.8765498652291106, v1 0.6115363881401618,
v2 0.038733153638814015. The pre-registered v2 bound (at most 0.10) held and v0 was higher than v2.
Per-hack values are in the results file.

### Hack-advantage sign change at group size 2 (SIMULATED, no model)

`scripts/sign_flip_by_reward.py` at G=2 (20 groups per task, mixing probability 0.5, seed 7, TEST
split) gives a mean v0-vs-v2 sign-change share of 0.6383885286791867, averaged over the seven hacks
(`results/293-sign-flip-group-size2.json`, `/sign_flip_share/v0_vs_v2/mean_share`). This is below
the G=4 value 0.8958503586543898 and above the pre-registered floor of 0.50, so the hypothesis held.
The mean strict-reversal share is 0.07579640463071512. Per-hack values are in the results file.

### Verifier error rate 0.3 (SIMULATED, iter 299)

At verifier error rate 0.3 the primary-cell false-positive minus false-negative exploit share gap is 0.22872611002107404 (95% interval 0.22358930395423438 to 0.23343205130323225), above the error-rate-0.2 value of 0.1653428751179616. Source: `results/299-verifier-error-rate03.json`; reproduce with `python3 scripts/verifier_error_asymmetry.py --split test --group-size 8 --groups-per-task 20 --error-rate 0.3 --seed 7 --out results/299-verifier-error-rate03.json`.

### Style-keyed verifier error (SIMULATED, iteration 333)

Simulated, no model. When the verifier's false-positive acceptance depends on an output style feature and the policy raises that feature's rate from 0.2 to 0.8, the exploit share of reinforced rollouts exceeds that of independent error at 0.2 by at least 0.2254 in every seed (`results/333-style-seeds.json`, `/style_dependent_error/exploit_share_style_minus_independent_min`). At the base style rate the difference is at most 0.0029 in absolute value (`/style_dependent_error/base_rate_diff_max_abs`).

### Task-persistent verifier error (SIMULATED, iteration 336)

Simulated, no model. A seeded 20% of tasks accept every wrong output and the rest accept none (marginal rate 0.2), versus independent error at 0.2. This is the same construction as the clustered run of iteration 329, re-registered under a new hypothesis; the pre-registered threshold was a zero-variance group share difference of at least 0.20 in every seed. The minimum over seeds 7, 13 and 21 was 0.1885 (`results/336-persistent-seeds.json`, `/task_persistent/zero_variance_share_diff_min`), so the hypothesis is refuted. The exploit share difference stayed at most 0.0031 in absolute value (`/task_persistent/exploit_share_diff_max_abs`).

```
python3 scripts/verifier_error_persistent_seeds.py --split test --group-size 8 --groups-per-task 20 --error-rate 0.2 --prob 0.5 --seeds 7 13 21 --out results/336-persistent-seeds.json
```

### Isomorphic-perturbation rung (SIMULATED, no model)

A scripted memorizer that replays the original-order gold items is accepted by the strict rung on
1.0 of the 265 TEST tasks, but by the strict-plus-reversed-order rung on 0.683 (survivors: 0.570
single-item tasks, 0.113 unanswerable tasks, 0.0 multi-item palindromes); the honest policy is
accepted on 1.0 under both. Pre-registered range 0.55 to 0.75, so the prediction held. This
probes one perturbation (item order) only.

```
python3 scripts/isomorphic_rung.py --split test --seed 7 --out results/360-isomorphic-rung.json
```

### Simulated GRPO audit-budget sweep (SIMULATED, no model)

`python3 scripts/simulated_grpo_audit_budget.py --seed 7 --out results/394-simulated-grpo-audit-budget.json`
runs a two-action softmax GRPO loop per task where the cheap check accepts exploit rollouts more
often than honest ones, and a strong verifier re-checks a fraction of exploit-accepted rollouts.
Mean final exploit probability by audit budget (0, 0.25, 0.5, 0.75, 1.0):
0.955, 0.606, 0.0553, 0.0214, 0.0131
(`/sweep/*` in `results/394-simulated-grpo-audit-budget.json`). The pre-registered hypotheses held
(0.606 at budget 0.25 against a floor of 0.50; 0.0131 at budget 1.0 against a ceiling of 0.05; non-increasing in budget).
This is a toy policy, not a trained model.

### Honest-accept sweep (simulated)

Simulated, model-free: with the exploit accept rate held at 0.6 and no audit, the smallest honest
accept rate on the grid at which mean final exploit probability falls below 0.50 is 0.5
(`results/405-honest-accept-sweep.json`, `/crossover/min_honest_accept_exploit_lt_0p5`).

```
PYTHONPATH=src python3 scripts/simulated_grpo_honest_accept_sweep.py --seed 7 --out results/405-honest-accept-sweep.json
```
