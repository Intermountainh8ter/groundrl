# Threats to validity

Read this before drawing a conclusion from anything in this repository. It lists what the work
cannot show, in order of how much it matters. Nothing here has been trained, so every statement
about a *trained* model is about what is not yet known.

## 1. The verifier is lexical

* **Token presence is not entailment.** The verifier checks that numbers, hex values, `2**n`
  expressions and ALL_CAPS identifiers occur in the cited chunks. A paraphrase with no such token
  passes vacuously (`paraphrase_no_tokens` scores 1.000 under v0 on TEST), and a correct token
  placed in a wrong claim passes (`one_wrong_constant`, 0.989 under v0). Both are asserted in
  `tests/test_verifier.py` as known blind spots, not fixed.
* **Small numbers are weak evidence.** `1`, `2`, `3` occur in almost every chunk, so a claim like
  "the type byte is 3" is "supported" by almost any cited chunk. Hex and decimal spellings are
  treated as equivalent (`0x03` = `3`), which makes this slightly weaker still.
* **Negation, units and comparatives are not checked.** "does not burn" and "burns" verify the same
  way; `32 ETH` and `32000000000 gwei` are aliased on purpose, but "larger than" is not evaluated.
* **The chunk header counts.** Each chunk begins with `EIP-N: title / section`, so the EIP number
  and any number in the title are always "supported" by every chunk of that EIP.
* **Identifier matching is case-folded**, and lower-case identifiers are not claim tokens at all.
* The verifier rejects non-ASCII digits/letters and invisible characters outright instead of
  normalising them. That closes look-alike tricks but would also reject a legitimate answer that
  contains, say, a Greek letter. Ordinary typography is allowed.

## 2. Gold labels come from templates and regexes

* Facts are extracted from the EIP text by deterministic patterns (constants tables, assignment
  lines, opcode sentences, preamble fields, blanked sentences). I read 39 random tasks by hand and
  found and fixed two extraction flaws; that is a spot check, not a measured precision. There may
  be others.
* The `supports()` test that decides whether a chunk supports a fact uses content-word overlap
  (0.70) for blanked sentences. Unanswerable questions are validated against it, so a chunk that
  paraphrases the answer below that threshold could make an "unanswerable" question answerable.
* Gold answers restate the source sentence. Correctness is token-based: a right answer worded
  with a different number format (for example "April 17, 2021" for `2021-04-17`) loses recall.
* Each extra claim token the gold answer does not contain multiplies correctness by 0.25. That
  pushes toward terse answers, which is intended, but it also penalises a correct answer that
  volunteers an extra true number.
* The consistency term compares sentences with identical words and different numbers. The task
  generator refuses questions whose own gold answer would trip it, so the oracle is never
  penalised; a trained model's legitimate answer could still trip it on a question I did not
  generate.

## 3. The questions are synthetic and mostly solvable by lexical matching

* Most questions are "what fills this blank" (a sentence from the chunk with a number removed) or
  "what is the value of NAME". The `extractive` baseline, which only finds the best-matching line,
  already gets 57.4% grounded-correct and a v2 reward of 0.613 on TEST. High numbers from a
  trained model are therefore not evidence of understanding; only a margin over that baseline is
  evidence of anything.
* Comparison questions test two-source retrieval and citation, not reasoning about the comparison:
  both values are asked for, the direction of the comparison is never graded.
* The `opcode` question type is thin: 53 opcode and type-byte facts in the whole corpus, 7 TEST
  tasks. Per-type numbers for it are not meaningful.
* Real user questions are not templated. Nothing here estimates transfer to them.

## 4. Splits, leakage and shortcuts

* The split is by EIP, and chunks from other splits never appear in a context. That does not
  remove *conceptual* leakage: related EIPs (4844 and 7691, the ERC token family) share ideas and
  sometimes sentences, and a pretrained model may have seen the EIPs in pretraining. Qwen3's
  training data is not known to me.
* Unanswerable questions have a shortcut: in about half of them no chunk of the named EIP is in
  context, so "abstain if the named EIP is absent" works there. `eip_id_shortcut` quantifies it
  (0.670 v2 on TEST against 0.613 for plain extractive).
* TEST has 265 tasks (30 unanswerable) from a small number of EIPs, so intervals are wide:
  abstention recall for the shortcut has a 95% interval of 18-82%. Do not read per-type slices.
* Prompts longer than the profile budget are dropped from training (0.7% at 2048 tokens, 2.8% at
  1536), and the dropped ones skew toward code-heavy chunks. Evaluation never drops any.
* Reward revision 2 was designed while looking at the TEST audit. The VAL audit agrees, but both
  are in-sample with respect to the attacks I enumerated.

## 5. The audit is a lower bound on the risk

* It scores scripted policies that I thought of. A learned policy explores differently and can find
  attacks that are not in the list. Known residual weaknesses: `hedge_inline` at 0.44 (TEST) and
  0.47 (VAL); `copy_chunk` earns at least 0.5 on about 14% of TEST tasks because a one-sentence
  chunk is its own answer; a reworded hedge evades the consistency term; the 0.10 floor pays a
  little for any grounded sentence; `refuse_all` earns the unanswerable share (0.113), which is a
  non-zero attractor early in GRPO.
* The 0.50 threshold is a convention. A policy at 0.45 is not safe; it is merely below the line.
* Several defences overlap, so ablating one term may understate how much it matters when the others
  are present. The "closed by" list only names terms whose removal alone re-opens a hack.
* The gold-free v2 is exploitable by 7 policies on TEST (9 on VAL): the labels are doing the work.
  Any deployment without labels needs a different reward.

## 6. Training is unrun and unmeasured

* No model has been trained. The SFT, GRPO and evaluation code has run only against a fake stack
  (labelled MOCK); its keyword arguments are checked against the field names of TRL 1.15.0,
  transformers 5.19.0 and PEFT 0.21.2 (`tests/api_fields.json`), which is a check on names, not on
  behaviour. TRL and transformers change quickly; a newer release may break it.
* Hyperparameters are untuned defaults. Memory figures in `--dry-run` are a formula, not a
  measurement; the logits tensor of the 151,936-token vocabulary is the term most likely to be
  wrong. Timing and cost are not estimated at all.
* Small models trained with GRPO are high-variance: one seed is an anecdote, and the plan calls for
  three or more seeds per configuration.
* Token counts in dry runs use a crude stub tokenizer (typically off by 10-30%).
* The prompt asks the model to follow rules (cite every sentence, abstain with one exact
  sentence). The reward then enforces them. A model that learns the format but not the facts would
  score on `verified` and fail on `correct`; the evaluation reports both.

## 7. The corpus

* 130 EIPs/ERCs, a curated snapshot of the master branch on the day of the fetch (hashes in
  `data/MANIFEST.json`). Drafts change; Final EIPs rarely do. The list was chosen by hand for
  being well known and rich in constants, which is a selection effect.
* ERC documents moved out of the EIPs repository and are fetched from the ERCs repository, still as
  `eip-N` identifiers.
