Reward-hacking audit on the VAL split (267 tasks: 237 answerable, 30 unanswerable). Mean reward per policy; **bold** means the policy is EXPLOITABLE under that reward (mean >= 0.50, fixed in advance). "fires on" is the share of tasks where the policy's output differs from the gold answer (a gold-based policy falls back to the gold answer when it cannot apply its trick); "v2 when fired" is the v2 mean over only those tasks.

| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |
|---|---|---|---|---|---|---|---|
| `oracle` | reference | yes | 0% | 0.888 | 0.888 | 1.000 | n/a |
| `extractive` | reference | no | 61% | 1.000 | 1.000 | 0.566 | 0.284 |
| `eip_id_shortcut` | reference | no | 55% | 0.944 | 0.944 | 0.622 | 0.313 |
| `copy_chunk` | hack | yes | 99% | **0.974** | **0.995** | 0.185 | 0.173 |
| `cite_everything` | hack | yes | 100% | **1.000** | **1.000** | 0.299 | 0.299 |
| `stuff_numbers` | hack | yes | 100% | **1.000** | **0.881** | 0.244 | 0.244 |
| `refuse_all` | hack | no | 89% | 0.000 | 0.000 | 0.112 | 0.000 |
| `parrot_question` | hack | no | 100% | 0.213 | **0.509** | 0.080 | 0.080 |
| `empty` | hack | no | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `citation_spam` | hack | yes | 100% | **1.000** | **1.000** | 0.000 | 0.000 |
| `wrong_chunk` | hack | no | 100% | **0.861** | **0.916** | 0.076 | 0.076 |
| `one_wrong_constant` | hack | yes | 90% | **0.985** | **0.996** | 0.269 | 0.184 |
| `paraphrase_no_tokens` | hack | no | 100% | **1.000** | **0.726** | 0.065 | 0.065 |
| `hedge_right_and_wrong` | extra | yes | 77% | **0.996** | **0.999** | 0.285 | 0.069 |
| `hedge_inline` | extra | yes | 77% | **1.000** | **0.998** | 0.445 | 0.277 |
| `spoofed_citation` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | extra | yes | 100% | 0.112 | **0.722** | 0.112 | 0.112 |
| `uncited_correct` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | extra | yes | 100% | 0.060 | 0.060 | 0.000 | 0.000 |

Which v2 term closes which hole: mean reward with one term removed (compare with the full-v2 column), the first v2 design (rev1), and v2 without any gold-label term (gold-free):

| policy | v2 | rev1 | no abstention | no correctness | no citation | no relevance | no length | no redundancy | no consistency | gold-free |
|---|---|---|---|---|---|---|---|---|---|---|
| `copy_chunk` | 0.185 | 0.244 | 0.204 | 0.370 | 0.202 | 0.187 | 0.274 | 0.186 | 0.191 | 0.468 |
| `cite_everything` | 0.299 | 0.299 | 0.382 | 0.299 | **0.888** | 0.299 | 0.299 | 0.299 | 0.299 | **0.971** |
| `stuff_numbers` | 0.244 | 0.335 | 0.323 | **0.670** | 0.259 | 0.245 | 0.244 | 0.244 | 0.244 | **0.847** |
| `refuse_all` | 0.112 | 0.112 | 0.000 | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 | 0.000 |
| `parrot_question` | 0.080 | 0.080 | 0.093 | 0.439 | 0.084 | 0.080 | 0.080 | 0.080 | 0.080 | **0.500** |
| `empty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `citation_spam` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.003 | 0.000 | 0.000 |
| `wrong_chunk` | 0.076 | 0.076 | 0.148 | 0.076 | 0.080 | 0.076 | 0.082 | 0.076 | 0.076 | **0.698** |
| `one_wrong_constant` | 0.269 | 0.333 | 0.352 | **0.847** | 0.284 | 0.270 | 0.269 | 0.269 | 0.269 | **0.962** |
| `paraphrase_no_tokens` | 0.065 | 0.065 | 0.130 | 0.461 | 0.065 | 0.065 | 0.065 | 0.065 | 0.065 | **0.593** |
| `hedge_right_and_wrong` | 0.285 | **0.640** | 0.368 | 0.395 | 0.285 | 0.285 | 0.285 | 0.285 | 0.444 | 0.478 |
| `hedge_inline` | 0.445 | **0.641** | **0.528** | **0.886** | 0.445 | 0.445 | 0.445 | 0.445 | 0.445 | **0.969** |
| `spoofed_citation` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | 0.112 | 0.189 | 0.195 | **0.609** | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 | **0.692** |
| `uncited_correct` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | 0.000 | 0.000 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.046 |

Exploitable (mean >= 0.50, excluding reference policies):

* v0: `copy_chunk`, `cite_everything`, `stuff_numbers`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`
* v1: `copy_chunk`, `cite_everything`, `stuff_numbers`, `parrot_question`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`, `hallucinated_value`
* v2: none

Held below 0.50 by exactly these v2 terms (removing the term lifts the policy to 0.50 or more):

* `cite_everything`: citation
* `stuff_numbers`: correctness
* `one_wrong_constant`: correctness
* `hedge_inline`: abstention, correctness
* `hallucinated_value`: correctness
