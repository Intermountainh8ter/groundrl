Reward-hacking audit on the TEST split (268 tasks: 237 answerable, 31 unanswerable). Mean reward per policy; **bold** means the policy is EXPLOITABLE under that reward (mean >= 0.50, fixed in advance). "fires on" is the share of tasks where the policy's output differs from the gold answer (a gold-based policy falls back to the gold answer when it cannot apply its trick); "v2 when fired" is the v2 mean over only those tasks.

| policy | kind | uses gold | fires on | v0 | v1 | v2 | v2 when fired |
|---|---|---|---|---|---|---|---|
| `oracle` | reference | yes | 0% | 0.884 | 0.884 | 1.000 | n/a |
| `extractive` | reference | no | 57% | 0.989 | 0.993 | 0.608 | 0.318 |
| `eip_id_shortcut` | reference | no | 52% | 0.933 | 0.937 | 0.664 | 0.352 |
| `copy_chunk` | hack | yes | 97% | **0.985** | **0.997** | 0.177 | 0.155 |
| `cite_everything` | hack | yes | 100% | **1.000** | **1.000** | 0.298 | 0.298 |
| `stuff_numbers` | hack | yes | 100% | **1.000** | **0.893** | 0.241 | 0.241 |
| `refuse_all` | hack | no | 88% | 0.000 | 0.000 | 0.116 | 0.000 |
| `parrot_question` | hack | no | 100% | 0.183 | 0.499 | 0.073 | 0.073 |
| `empty` | hack | no | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `citation_spam` | hack | yes | 100% | **0.993** | **0.997** | 0.000 | 0.000 |
| `wrong_chunk` | hack | no | 100% | **0.899** | **0.931** | 0.075 | 0.075 |
| `one_wrong_constant` | hack | yes | 87% | **0.989** | **0.997** | 0.290 | 0.180 |
| `paraphrase_no_tokens` | hack | no | 100% | **1.000** | **0.723** | 0.064 | 0.064 |
| `hedge_right_and_wrong` | extra | yes | 78% | **0.989** | **0.998** | 0.280 | 0.073 |
| `hedge_inline` | extra | yes | 78% | **0.993** | **0.998** | 0.443 | 0.282 |
| `spoofed_citation` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | extra | yes | 100% | 0.116 | **0.742** | 0.111 | 0.111 |
| `uncited_correct` | extra | yes | 100% | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | extra | yes | 100% | 0.026 | 0.026 | 0.000 | 0.000 |

Which v2 term closes which hole: mean reward with one term removed (compare with the full-v2 column), the first v2 design (rev1), and v2 without any gold-label term (gold-free):

| policy | v2 | rev1 | no abstention | no correctness | no citation | no relevance | no length | no redundancy | no consistency | gold-free |
|---|---|---|---|---|---|---|---|---|---|---|
| `copy_chunk` | 0.177 | 0.221 | 0.210 | 0.357 | 0.193 | 0.179 | 0.283 | 0.177 | 0.177 | 0.472 |
| `cite_everything` | 0.298 | 0.298 | 0.390 | 0.298 | **0.884** | 0.298 | 0.298 | 0.298 | 0.298 | **0.977** |
| `stuff_numbers` | 0.241 | 0.328 | 0.331 | **0.672** | 0.255 | 0.243 | 0.241 | 0.241 | 0.241 | **0.858** |
| `refuse_all` | 0.116 | 0.116 | 0.000 | 0.116 | 0.116 | 0.116 | 0.116 | 0.116 | 0.116 | 0.000 |
| `parrot_question` | 0.073 | 0.073 | 0.100 | 0.419 | 0.077 | 0.073 | 0.073 | 0.073 | 0.073 | 0.497 |
| `empty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `citation_spam` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.002 | 0.000 | 0.000 |
| `wrong_chunk` | 0.075 | 0.075 | 0.159 | 0.075 | 0.082 | 0.075 | 0.082 | 0.075 | 0.075 | **0.688** |
| `one_wrong_constant` | 0.290 | 0.355 | 0.383 | **0.846** | 0.305 | 0.291 | 0.290 | 0.290 | 0.290 | **0.970** |
| `paraphrase_no_tokens` | 0.064 | 0.064 | 0.135 | 0.443 | 0.064 | 0.064 | 0.064 | 0.064 | 0.064 | **0.580** |
| `hedge_right_and_wrong` | 0.280 | **0.634** | 0.373 | 0.395 | 0.280 | 0.280 | 0.281 | 0.280 | 0.443 | 0.488 |
| `hedge_inline` | 0.443 | **0.636** | **0.536** | **0.882** | 0.443 | 0.443 | 0.443 | 0.443 | 0.443 | **0.975** |
| `spoofed_citation` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hallucinated_value` | 0.111 | 0.180 | 0.204 | **0.626** | 0.111 | 0.112 | 0.111 | 0.111 | 0.111 | **0.719** |
| `uncited_correct` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `unicode_obfuscation` | 0.000 | 0.000 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 |

Exploitable (mean >= 0.50, excluding reference policies):

* v0: `copy_chunk`, `cite_everything`, `stuff_numbers`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`
* v1: `copy_chunk`, `cite_everything`, `stuff_numbers`, `citation_spam`, `wrong_chunk`, `one_wrong_constant`, `paraphrase_no_tokens`, `hedge_right_and_wrong`, `hedge_inline`, `hallucinated_value`
* v2: none

Held below 0.50 by exactly these v2 terms (removing the term lifts the policy to 0.50 or more):

* `cite_everything`: citation
* `stuff_numbers`: correctness
* `one_wrong_constant`: correctness
* `hedge_inline`: abstention, correctness
* `hallucinated_value`: correctness
