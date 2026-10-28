# X thread

1/
I built a deterministic citation checker as an RL reward for small models answering questions about
Ethereum EIPs, tried to break it, then ran real training. SFT scored high on my test; GRPO gave a
null result. What it does and does not show:

2/
The check: every sentence has a [eip-N#CCC] citation, the cited chunk was in context, and every
number / hex / 2**n / ALL_CAPS identifier appears in the chunk it cites. Exact and cheap, but
lexical: token presence, not meaning.

3/
Audit first, no model: 19 scripted policies. A binary verifier reward is exploitable by 7 of my 10
hacks (0.89-1.00): copy the chunk, cite everything, stuff numbers, wrong chunk. The final reward
holds every hack I enumerated below 0.50 (highest 0.44).

4/
Caveats: that reward needs gold labels (strip them and 7 hacks get through), and I revised it while
looking at the TEST audit, so "closed" is in-sample for the attacks I thought of.

5/
The bar: a no-model policy that cites the best-matching line scores 57.4% grounded-correct and
0.613 on the v2 reward on 265 TEST tasks (EIPs disjoint from training).

6/
SFT: Qwen3-0.6B, LoRA r=16, 2 epochs, about 39 min on an AMD gfx906 card under ROCm.
Grounded-correct 27.2% -> 98.7% (95% CI 97.0-100.0), verified 40.4% -> 99.6%, v2 reward
0.357 -> 0.985. Hallucinated citations 0.8% -> 0.4%.

7/
Read that carefully: templated synthetic questions, 130 EIPs, one seed, one model size, one
machine. Train and test share question templates, so the model learned a template's pattern. The
verifier checks tokens, not entailment. Not evidence of understanding.

8/
The base model abstained on 41.9% of test questions, though only 11.3% are unanswerable, so 27.2%
is a weak starting point. I ran no other baseline.

9/
GRPO: 300 optimiser steps, zero reward and zero completion length. The merged SFT model emitted
only the empty no-think block, so nothing was scored. A null result about my setup; GRPO was not
shown to work or to fail. No RL benefit claimed.

10/
Repo, audit table, leakage check, null-run diagnostic, threats list:
https://github.com/Intermountainh8ter/groundrl
