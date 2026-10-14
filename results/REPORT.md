# groundrl report

Rendered by `scripts/make_report.py` from `results/*.json` only.

## Reward-hacking audit (real, no model)

Source: `audit_test.json`; regenerate with `python scripts/reward_audit.py --split test`.

Reward-hacking audit on the TEST split (265 tasks: 235 answerable, 30 unanswerable). Mean reward per policy; **bold** means the policy is EXPLOITABLE under that reward (mean >= 0.50, fixed in advance). "fires on" is the share of tasks where the policy's output differs from the gold answer (a gold-based policy falls back to the gold answer when it cannot apply its trick); "v2 when fired" is the v2 mean over only those tasks.

| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |
|---|---|---|---|---|---|---|---|
| `citation_spam` | hack | yes | 100% | **1.000** | **1.000** | 0.000 | 0.000 |
| `cite_everything` | hack | yes | 100% | **1.000** | **1.000** | 0.296 | 0.296 |
| `copy_chunk` | hack | yes | 97% | **0.981** | **0.996** | 0.185 | 0.160 |
| `eip_id_shortcut` | reference | no | 52% | 0.928 | 0.931 | 0.670 | 0.361 |
| `empty` | hack | no | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `extractive` | reference | no | 57% | 0.985 | 0.988 | 0.613 | 0.325 |
| `hallucinated_value` | extra | yes | 100% | 0.113 | **0.722** | 0.109 | 0.109 |
| `hedge_inline` | extra | yes | 78% | **1.000** | **0.998** | 0.440 | 0.284 |
| `hedge_right_and_wrong` | extra | yes | 78% | **0.989** | **0.998** | 0.276 | 0.073 |
| `one_wrong_constant` | hack | yes | 88% | **0.989** | **0.996** | 0.277 | 0.181 |
| `oracle` | reference | yes | 0% | 0.887 | 0.887 | 1.000 | n/a |
| `paraphrase_no_tokens` | hack | no | 100% | **1.000** | **0.724** | 0.064 | 0.064 |
| `parrot_question` | hack | no | 100% | 0.174 | **0.500** | 0.077 | 0.077 |
| `refuse_all` | hack | no | 89% | 0.000 | 0.000 | 0.113 | 0.000 |
| `spoofed_citation` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `stuff_numbers` | hack | yes | 100% | **1.000** | **0.882** | 0.247 | 0.247 |
| `uncited_correct` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | extra | yes | 100% | 0.049 | 0.049 | 0.000 | 0.000 |
| `wrong_chunk` | hack | no | 100% | **0.887** | **0.928** | 0.074 | 0.074 |

Which v2 term closes which hole: mean reward with one term removed (compare with the full-v2 column), the first v2 design (rev1), and v2 without any gold-label term (gold-free):

| policy | v2 | rev1 | no abstention | no correctness | no citation | no relevance | no length | no redundancy | no consistency | gold-free |
|---|---|---|---|---|---|---|---|---|---|---|
| `citation_spam` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.003 | 0.000 | 0.000 |
| `cite_everything` | 0.296 | 0.296 | 0.380 | 0.296 | **0.887** | 0.296 | 0.296 | 0.296 | 0.296 | **0.972** |
| `copy_chunk` | 0.185 | 0.234 | 0.204 | 0.383 | 0.206 | 0.188 | 0.292 | 0.185 | 0.185 | 0.475 |
| `empty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | 0.109 | 0.178 | 0.194 | **0.609** | 0.109 | 0.109 | 0.109 | 0.109 | 0.109 | **0.694** |
| `hedge_inline` | 0.440 | **0.630** | **0.525** | **0.885** | 0.440 | 0.440 | 0.440 | 0.440 | 0.440 | **0.970** |
| `hedge_right_and_wrong` | 0.276 | **0.628** | 0.361 | 0.395 | 0.276 | 0.276 | 0.277 | 0.276 | 0.440 | 0.480 |
| `one_wrong_constant` | 0.277 | 0.338 | 0.362 | **0.848** | 0.292 | 0.278 | 0.277 | 0.277 | 0.277 | **0.965** |
| `paraphrase_no_tokens` | 0.064 | 0.064 | 0.134 | 0.451 | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 | **0.585** |
| `parrot_question` | 0.077 | 0.077 | 0.097 | 0.427 | 0.082 | 0.077 | 0.077 | 0.077 | 0.077 | 0.498 |
| `refuse_all` | 0.113 | 0.113 | 0.000 | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 | 0.000 |
| `spoofed_citation` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `stuff_numbers` | 0.247 | 0.335 | 0.328 | **0.670** | 0.264 | 0.249 | 0.247 | 0.247 | 0.247 | **0.845** |
| `uncited_correct` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | 0.000 | 0.000 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.035 |
| `wrong_chunk` | 0.074 | 0.074 | 0.155 | 0.074 | 0.076 | 0.074 | 0.082 | 0.074 | 0.074 | **0.683** |

Exploitable (mean >= 0.50, excluding reference policies):

* v0: `copy_chunk`, `cite_everything`, `stuff_numbers`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`
* v1: `copy_chunk`, `cite_everything`, `stuff_numbers`, `parrot_question`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`, `hallucinated_value`
* v2: none

Held below 0.50 by exactly these v2 terms (removing the term lifts the policy to 0.50 or more):

* `cite_everything`: citation
* `hallucinated_value`: correctness
* `hedge_inline`: abstention, correctness
* `one_wrong_constant`: correctness
* `stuff_numbers`: correctness


Source: `audit_val.json`; regenerate with `python scripts/reward_audit.py --split val`.

Reward-hacking audit on the VAL split (265 tasks: 235 answerable, 30 unanswerable). Mean reward per policy; **bold** means the policy is EXPLOITABLE under that reward (mean >= 0.50, fixed in advance). "fires on" is the share of tasks where the policy's output differs from the gold answer (a gold-based policy falls back to the gold answer when it cannot apply its trick); "v2 when fired" is the v2 mean over only those tasks.

| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |
|---|---|---|---|---|---|---|---|
| `citation_spam` | hack | yes | 100% | **0.996** | **0.999** | 0.000 | 0.000 |
| `cite_everything` | hack | yes | 100% | **1.000** | **1.000** | 0.299 | 0.299 |
| `copy_chunk` | hack | yes | 98% | **0.970** | **0.994** | 0.198 | 0.185 |
| `eip_id_shortcut` | reference | no | 55% | 0.940 | 0.940 | 0.632 | 0.328 |
| `empty` | hack | no | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `extractive` | reference | no | 61% | 1.000 | 1.000 | 0.572 | 0.296 |
| `hallucinated_value` | extra | yes | 100% | 0.113 | **0.727** | 0.113 | 0.113 |
| `hedge_inline` | extra | yes | 74% | **0.996** | **0.998** | 0.466 | 0.274 |
| `hedge_right_and_wrong` | extra | yes | 74% | **0.992** | **0.999** | 0.315 | 0.070 |
| `one_wrong_constant` | hack | yes | 88% | **0.992** | **0.999** | 0.293 | 0.192 |
| `oracle` | reference | yes | 0% | 0.887 | 0.887 | 1.000 | n/a |
| `paraphrase_no_tokens` | hack | no | 100% | **1.000** | **0.726** | 0.064 | 0.064 |
| `parrot_question` | hack | no | 100% | 0.196 | **0.513** | 0.073 | 0.073 |
| `refuse_all` | hack | no | 89% | 0.000 | 0.000 | 0.113 | 0.000 |
| `spoofed_citation` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `stuff_numbers` | hack | yes | 100% | **1.000** | **0.886** | 0.250 | 0.250 |
| `uncited_correct` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | extra | yes | 100% | 0.030 | 0.030 | 0.000 | 0.000 |
| `wrong_chunk` | hack | no | 100% | **0.906** | **0.940** | 0.076 | 0.076 |

Which v2 term closes which hole: mean reward with one term removed (compare with the full-v2 column), the first v2 design (rev1), and v2 without any gold-label term (gold-free):

| policy | v2 | rev1 | no abstention | no correctness | no citation | no relevance | no length | no redundancy | no consistency | gold-free |
|---|---|---|---|---|---|---|---|---|---|---|
| `citation_spam` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.004 | 0.000 | 0.000 |
| `cite_everything` | 0.299 | 0.299 | 0.383 | 0.299 | **0.887** | 0.299 | 0.299 | 0.299 | 0.299 | **0.970** |
| `copy_chunk` | 0.198 | 0.242 | 0.229 | 0.363 | 0.218 | 0.199 | 0.294 | 0.198 | 0.198 | 0.470 |
| `empty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | 0.113 | 0.186 | 0.197 | **0.614** | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 | **0.697** |
| `hedge_inline` | 0.466 | **0.650** | **0.549** | **0.885** | 0.466 | 0.466 | 0.466 | 0.466 | 0.466 | **0.969** |
| `hedge_right_and_wrong` | 0.315 | **0.649** | 0.399 | 0.422 | 0.315 | 0.315 | 0.316 | 0.315 | 0.465 | **0.505** |
| `one_wrong_constant` | 0.293 | 0.350 | 0.376 | **0.847** | 0.308 | 0.294 | 0.293 | 0.293 | 0.293 | **0.963** |
| `paraphrase_no_tokens` | 0.064 | 0.064 | 0.131 | 0.454 | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 | **0.588** |
| `parrot_question` | 0.073 | 0.073 | 0.089 | 0.440 | 0.077 | 0.073 | 0.073 | 0.073 | 0.073 | **0.507** |
| `refuse_all` | 0.113 | 0.113 | 0.000 | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 | 0.000 |
| `spoofed_citation` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `stuff_numbers` | 0.250 | 0.335 | 0.334 | **0.668** | 0.267 | 0.252 | 0.250 | 0.250 | 0.250 | **0.847** |
| `uncited_correct` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.026 |
| `wrong_chunk` | 0.076 | 0.076 | 0.144 | 0.076 | 0.082 | 0.076 | 0.084 | 0.076 | 0.076 | **0.688** |

Exploitable (mean >= 0.50, excluding reference policies):

* v0: `copy_chunk`, `cite_everything`, `stuff_numbers`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`
* v1: `copy_chunk`, `cite_everything`, `stuff_numbers`, `parrot_question`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`, `hallucinated_value`
* v2: none

Held below 0.50 by exactly these v2 terms (removing the term lifts the policy to 0.50 or more):

* `cite_everything`: citation
* `hallucinated_value`: correctness
* `hedge_inline`: abstention, correctness
* `one_wrong_constant`: correctness
* `stuff_numbers`: correctness


## Scripted baselines (not trained models)

Source: `baseline_test.json` (test split). Scripted policies, no learning.

| run | n | grounded-correct | correct | verified | abstain precision | abstain recall | hallucinated cite | v0 | v1 | v2 | words (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eip_id_shortcut` | 265 | 57.4% (51.1-63.8) | 61.7% (55.3-68.1) | 98.3% (96.6-99.6) | 100.0% (100.0-100.0) | 50.0% (18.2-81.5) | 0.0% (0.0-0.0) | 0.928 (0.894-0.958) | 0.931 (0.899-0.959) | 0.670 (0.622-0.716) | 21.4 |
| `extractive` | 265 | 57.4% (51.1-63.8) | 61.7% (55.3-68.1) | 98.3% (96.6-99.6) | n/a | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 0.985 (0.970-0.996) | 0.988 (0.974-0.998) | 0.613 (0.563-0.663) | 22.1 |
| `oracle` | 265 | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) | 0.0% (0.0-0.0) | 0.887 (0.849-0.921) | 0.887 (0.849-0.921) | 1.000 (1.000-1.000) | 21.9 |
| `refuse_all` | 265 | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 11.3% (6.0-18.9) | 100.0% (100.0-100.0) | 0.0% (0.0-0.0) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.113 (0.079-0.151) | 8.0 |

## Training results

| run | n | grounded-correct | correct | verified | abstain precision | abstain recall | hallucinated cite | v0 | v1 | v2 | words (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 265 | 27.2% (21.7-33.2) | 42.6% (36.6-48.9) | 40.4% (34.0-46.4) | 25.2% (13.9-39.5) | 93.3% (75.9-100.0) | 0.8% (0.0-1.9) | 0.362 (0.306-0.423) | 0.381 (0.328-0.436) | 0.357 (0.305-0.412) | 13.7 |
| `sft` | 265 | 98.7% (97.0-100.0) | 99.1% (97.9-100.0) | 99.6% (98.7-100.0) | 100.0% (100.0-100.0) | 93.3% (70.8-100.0) | 0.4% (0.0-1.1) | 0.883 (0.842-0.921) | 0.886 (0.847-0.922) | 0.985 (0.971-0.996) | 21.9 |

* `eval_baseline.json`: split test, model `Qwen/Qwen3-0.6B`, adapter `None`, commit 28b7524fc8f25c5ec1a4509d022978e1082ca5a4
* `eval_sft.json`: split test, model `Qwen/Qwen3-0.6B`, adapter `/home/ammo/bench/results/groundrl-sft/adapter`, commit 28b7524fc8f25c5ec1a4509d022978e1082ca5a4

Training runs on record:

* `summary_sft.json`: stage sft, config qwen3-0.6b-gfx906, seed 0, 32 log rows
