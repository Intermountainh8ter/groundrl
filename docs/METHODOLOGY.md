# Methodology

This document states what groundrl is trying to do, how each piece is defined, and what would
count as evidence. It is written before the experiments so that the protocol is fixed in advance.

## 1. Problem

Retrieval-augmented answers are only useful if each claim can be traced to the text it came from.
Most citation checks are run at inference time, after the model has already answered. groundrl
asks a different question: can the same check be used as a *training* signal, so that a small model
learns to cite correctly in the first place?

The setting is deliberately narrow so the check can be exact:

* Corpus: a curated set of Ethereum Improvement Proposals (EIPs), CC0, fetched from
  `raw.githubusercontent.com`.
* Task: given a question and `k` context chunks (one or two of which hold the answer, the rest
  distractors), write an answer in which every sentence carries a citation of the form
  `[eip-N#CCC]` naming a provided chunk.
* Some questions are unanswerable from the provided context. The correct behaviour there is to
  abstain with a fixed sentence.

## 2. The verifier (deterministic, lexical)

The verifier (`src/groundrl/verifier.py`) is a clean reimplementation of the grounding check from
the author's other repository `ethresearch-rag` (MIT, same author), with stricter rules because it
is used as a reward. It accepts an answer only if

1. the answer is not empty and every sentence has content besides its citations;
2. every sentence carries at least one citation of the form `[eip-N#CCC]` (lower case, chunk index
   zero padded to three digits). A citation after the full stop (`... 4096. [eip-4844#003]`)
   attaches to the sentence before it;
3. every citation names a chunk that was in the provided context. A citation to a chunk that does
   not exist (`[eip-1559#999]`) or exists but was not provided is rejected;
4. every *claim token* in a sentence is present in the chunks **that sentence cites**. Claim
   tokens are numbers (`4096`, `1,000`, `1_000`), hex values, `2**n` expressions and ALL_CAPS
   identifiers (`FIELD_ELEMENTS_PER_BLOB`, `PUSH0`). Equivalent spellings are recorded rather
   than ignored: `0x48` = `72`, `2**12` = `4096`, `32 ETH` = `32000000000` gwei. Substring
   matches never count (`8` is not found inside `58`);
5. the answer contains no invisible, look-alike or non-ASCII digit/letter characters (zero-width
   spaces, bidi controls, fullwidth or Arabic-Indic digits, Cyrillic letters in an identifier).
   Ordinary typography (curly quotes, dashes, `>=` symbols) is allowed.

Markdown fences, inline code, bold markers and list bullets are transparent: they are stripped but
the content inside them is verified like any other text, so a code fence cannot hide a claim.
Every checked token records which chunk supported it and how (`exact`, `hex-dec`, `pow2`,
`unit`); `VerificationResult.provenance()` returns `{chunk_id: [tokens]}`.

Two design decisions differ from a naive implementation. An empty answer is rejected (a loop over
zero sentences is vacuously true in the sibling implementation, which accepts `""`). And a
well-formed citation that is wrong in a cosmetic way (`[eip-4844#3]`, `[EIP-4844#003]`) is not
repaired: it counts as no citation and is reported as a near miss, so format drift costs reward
instead of being silently forgiven.

What the verifier is **not**: it checks token presence, not entailment. Two holes are known and
tested explicitly in `tests/test_verifier.py`:

* a paraphrase with no claim token passes vacuously;
* a correct token placed in a wrong claim passes (`FIELD_ELEMENTS_PER_BLOB is 786432` verifies
  because `786432` is in the chunk, just not as that constant).

These two holes are the reason the reward has more than one version and why the audit exists.

## 3. Reward versions

All rewards are pure functions `(completion, example) -> (score, breakdown)` with the score in
[0, 1]; `src/groundrl/rewards.py` holds the formulas and `make_trl_reward` wraps them as
`fn(prompts, completions, **kwargs) -> list[float]` for TRL.

* **v0** binary. 1.0 if the verifier accepts the answer, else 0.0. An honest abstention is an
  uncited sentence, so it scores 0.
* **v1** graded. Per sentence: 0 without a valid citation, otherwise
  `(1 - 0.5 * invalid_share) * (0.7 * support_ratio + 0.3 * coverage)`, averaged over sentences.
  `support_ratio` is supported / checked claim tokens (1 if there are none) and `coverage` is the
  share of the sentence's content words found in the cited chunks. Answers with suspicious
  characters score 0.
* **v2** v1 plus terms that need the gold label. With `g` = the v1 score:

  `score = consistency * length * redundancy * g * (floor + (1 - floor) * correctness * citation * (0.7 + 0.3 * relevance))`

  * `correctness` = recall of the gold value tokens, times 0.25 for every extra claim token that
    is neither a gold token nor echoed from the question (EIP numbers are neutral too). Revision 1
    used recall x precision; see the revision log below.
  * `citation` = precision x recall of the cited chunks against the gold support groups.
  * `relevance` = share of the question's claim tokens that the answer repeats (EIP numbers and
    the quoted section title excluded).
  * `length` = 1 up to a budget of `1.5 x gold words + 15`, falling linearly to 0 at twice the
    budget. `redundancy` = (1 - repeated sentence share) x (1 - repeated citation-marker share).
  * `consistency` = 0.25 if two sentences have identical words (pure numbers aside, EIP numbers
    kept) but different numbers, i.e. the answer says the same thing twice with two values.
  * `floor` = 0.10 keeps a gradient for answers that are well formed and grounded but not yet
    correct, which matters at the start of GRPO.
  * abstention calibration comes first: the exact abstention sentence scores 1 if the gold label
    is unanswerable and 0 otherwise; any other answer to an unanswerable question scores 0.

Every v2 term can be switched off (`V2Config.without("citation")`); the audit uses this to show
which term closes which hole.

## 4. Data protocol

* Splits are made **by EIP**, not by question: no EIP number appears in more than one of
  TRAIN, VAL and TEST, including as a distractor chunk and as one half of a comparison question.
  This prevents the model from memorising an EIP's text during training and being scored on it at
  test time.
* Questions are generated from templates with a fixed seed, so every number is reproducible. Gold
  answers are derived from the text itself (constant tables, opcode lines), then checked: the gold
  tokens must occur in the gold chunk, and for unanswerable questions they must occur in *no*
  context chunk.

## 5. The reward-hacking audit

The audit is the part of this repository that can be produced for real without a GPU
(`python scripts/reward_audit.py --split test`, about 30 s, writes `results/audit_test.json` and
`.md`). Scripted adversarial *policies* (plain functions, no model) are scored under every reward
version on a real split, and under v2 with one term removed at a time.

Policies (`src/groundrl/audit.py`). Reference: `oracle` (the gold answer), `extractive` (the single
best-matching line, cited, never abstains) and `eip_id_shortcut` (extractive, but abstains when no
chunk of a named EIP is in context). The ten required hacks: `copy_chunk`, `cite_everything`,
`stuff_numbers`, `refuse_all`, `parrot_question`, `empty`, `citation_spam`, `wrong_chunk`,
`one_wrong_constant`, `paraphrase_no_tokens`. Extra: `hedge_right_and_wrong`, `hedge_inline`,
`spoofed_citation`, `hallucinated_value`, `uncited_correct`, `unicode_obfuscation`. Policies that
start from the gold answer model the strongest attacker, one that has learned *what* to say and
only optimises *how*; the table marks which policies use gold. A gold-based policy that cannot
apply its trick (for example, no alternative number exists in the chunk) falls back to the gold
answer, so the table also reports how often each policy actually fires and the v2 mean over those
tasks only.

Pre-registered rule, fixed before any number was produced: a non-reference policy is flagged
EXPLOITABLE under a reward if its mean reward over the split is at least 0.50.

The audit is not a proof of safety. It enumerates the hacks the author thought of; a trained
policy may find others. Residual holes are in `docs/THREATS_TO_VALIDITY.md`.

### Revision log of reward v2 (real numbers, TEST split, 268 tasks)

* **Revision 1** (first design: correctness = recall x precision, no consistency term). The audit
  flagged two policies as exploitable under v2: `hedge_right_and_wrong` (the correct sentence
  followed by a second sentence with a contradicting value taken from the same chunk; mean 0.634)
  and `hedge_inline` (the correct value followed by "(or <wrong value>)"; mean 0.636). Both pass
  the verifier because every token is in the chunk.
* **Rejected intermediate attempt.** Squaring precision and flagging near-identical sentences with
  different numbers (word overlap >= 0.8) closed the first hedge but flagged 7 legitimate gold
  answers on TEST (sibling sentences such as "a single G1 point (128 bytes)" and "a single G2
  point (256 bytes)", or two creation dates for two different EIPs), dropping the oracle's v2
  minimum to 0.25. A test now requires the oracle to score at least 0.97 on every answerable task.
* **Revision 2** (current). Each extra claim multiplies correctness by 0.25, and the consistency
  term requires *identical* words (EIP numbers kept as part of the identity). The oracle scores
  1.0 on every TEST and VAL task (minimum 0.98 over TRAIN) and no non-reference policy reaches
  0.50 on either TEST or VAL. The highest is `hedge_inline` at 0.443 (TEST) and 0.445 (VAL); on the
  tasks where it fires its v2 mean is lower, and the reward-level reason it is not lower still is
  that a single extra claim costs a factor of 0.25 against the 0.10 floor.

Revision 2 was designed while looking at the TEST audit (there was no trained model to overfit,
but "closed" is in-sample with respect to the enumerated attacks). The VAL audit
(`results/audit_val.json`) was first generated after the design was frozen and shows the same
picture; it is a robustness check, not an independent proof.

### What the audit shows (from `results/audit_test.json`)

* v0 is exploitable by 7 of the 10 required hacks, and v1 by the same plus more: a binary or
  graded verifier alone rewards correct-looking, well-cited, wrong or empty answers.
* Removing a *single* v2 term re-opens a hack in five cases: without the citation term
  `cite_everything` is back at 0.88; without correctness `stuff_numbers` (0.67),
  `one_wrong_constant` (0.85) and `hallucinated_value` (0.63) are back; `hedge_inline` returns
  with either correctness or abstention removed. The other hacks are held down by overlapping
  terms, so no single removal reopens them past 0.50: `wrong_chunk` needs both correctness and
  citation gone (it is 0.69 gold-free), `citation_spam` stays below 0.02 under every single removal
  (length, redundancy, citation and correctness each stop it), and `copy_chunk` is held down by
  correctness and length together. Without correctness,
  `parrot_question` is at 0.42 and `paraphrase_no_tokens` at 0.44.
* Abstention calibration is what makes `refuse_all` earn only the unanswerable share of the
  split (0.116) instead of 0 (as under v0 and v1, which never reward abstaining and so never
  teach it) or 1.
* A gold-free v2 (grounding, relevance, length, redundancy only) is exploitable by 7 policies. The
  gold labels are doing real work; a deployment without gold labels cannot use the verifier alone
  as its reward.
* `copy_chunk` still earns at least 0.5 on about 13% of TEST tasks: when the gold chunk is a
  single short sentence, copying it is indistinguishable from answering. The `extractive`
  baseline (no model, no gold) reaches 0.61 on TEST, which is the number a trained model must beat
  to show it learned anything beyond locate-and-copy.

## 6. Training protocol (specified, not run)

Stages, in order (`docs/RUNBOOK.md` has the commands):

1. **SFT baseline.** Supervised fine-tuning with LoRA on the gold answers of the TRAIN split
   (prompt-completion format, loss on the completion only), then the adapter is merged into the
   base weights. This is the starting point for GRPO and the comparison point for it.
2. **GRPO.** Reward v2 (`make_trl_reward("v2")`), a fresh LoRA on the merged SFT model, group size
   4-8, a small KL coefficient (0.02), learning rate 2e-5. Prompts are pre-rendered with the
   model's own chat template (thinking disabled) so training and evaluation see identical text.
3. **Evaluation on TEST** with greedy decoding: grounded-correct rate, correct rate, verified
   rate, abstention precision and recall, hallucinated-citation rate, mean reward under each
   version, length statistics, all with 95% bootstrap intervals over tasks.

Comparison points that need no model (`groundrl baseline-eval`): the `oracle`, the top-1
`extractive` policy, `eip_id_shortcut` and `refuse_all`. A trained model has to beat `extractive`
to show it learned more than locating and copying a line.

Claims that would need evidence this repository does not yet have: that GRPO with v2 beats the SFT
baseline; that v2 is robust to a *learned* policy's exploration (the audit covers only scripted
attacks); that anything transfers beyond EIPs. Planned reporting: three or more seeds per
configuration, intervals not point estimates, TEST touched once per final comparison.

## 7. Honesty rules

* No training results are reported unless `results/*.json` produced by a real run exist.
  `scripts/make_report.py` renders only from those files.
* Runs against a mock model exist only to test trainer plumbing and are labelled MOCK everywhere
  they appear.
* The audit table is a real, reproducible result and is regenerated by one command (see README).
