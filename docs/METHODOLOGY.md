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

  `score = length * redundancy * g * (floor + (1 - floor) * correctness * citation * (0.7 + 0.3 * relevance))`

  * `correctness` = recall x precision of the answer's claim tokens against the gold tokens. Tokens
    that also appear in the question, and the EIP numbers involved, are neutral.
  * `citation` = precision x recall of the cited chunks against the gold support groups.
  * `relevance` = share of the question's claim tokens that the answer repeats.
  * `length` = 1 up to a budget of `1.5 x gold words + 15`, falling linearly to 0 at twice the
    budget. `redundancy` = (1 - repeated sentence share) x (1 - repeated citation-marker share).
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

The audit is the part of this repository that can be produced for real without a GPU. A set of
scripted adversarial "policies" (no model involved) is scored under each reward version on the
real TEST split. A policy that earns a high reward without answering the question is a hack. The
audit reports, for every policy and every reward version, the mean reward and the fraction of
questions scoring at least 0.5. The comparison against an oracle policy (the gold answer) puts the
numbers in context.

The audit is not a proof of safety. It enumerates the hacks the author thought of; a trained
policy may find others. Residual holes are listed in `docs/THREATS_TO_VALIDITY.md`.

## 6. Honesty rules

* No training results are reported unless `results/*.json` produced by a real run exist.
  `scripts/make_report.py` renders only from those files.
* Runs against a mock model exist only to test trainer plumbing and are labelled MOCK everywhere
  they appear.
* The audit table is a real, reproducible result and is regenerated by one command (see README).
