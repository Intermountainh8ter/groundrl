# Train/test leakage check

**Date:** 2026-10-14
**Dataset:** `data/tasks/{train,val,test}.jsonl`
**Split policy:** per `src/groundrl/tasks.py:1`, "Seeded task generation with strict train/val/test separation by EIP."

## Measured overlap (test vs. train)

| metric | overlap | train | test |
|---|---|---|---|
| fact-id (`gold_facts[].id`) | 0 | 902 | 188 |
| document (EIP) | 0 | 91 | 19 |
| exact-question string | 0 / 265 | — | — |
| question *shape* (EIP name masked) | 8 / 238 (3.4%) | — | — |
| `gold_answer` string | 1 / 214 (0.5%) | — | — |

## Interpretation

Facts and documents are strictly held out — no EIP appears in both splits and no gold fact is reused.
The 3.4% question-shape overlap reflects the finite question-template set (e.g. "What is the
creation date of EIP-X?") applied to different EIPs; this is *not* leakage. The one gold-answer
collision is coincidental (a shared literal between two unrelated facts).

**Conclusion:** the SFT→test gain is a legitimate held-out improvement, not memorisation.
