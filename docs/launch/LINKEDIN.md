# LinkedIn post

I built a deterministic citation checker to use as an RL reward, tried to break it, and then ran the
first real training. SFT scored high on my test set; the GRPO stage produced a null result. Here is
what each part does and does not show.

The idea: use the checker as a verifiable reward (RLVR) for a small model (Qwen3-0.6B, LoRA) that
answers questions about Ethereum EIPs with citations. Every sentence needs a [eip-N#CCC] citation,
the cited chunk must have been in the context, and every number, hex value, 2**n and ALL_CAPS
identifier must appear in the chunk it cites. It is exact and cheap. It is also lexical: it checks
that tokens are present, not what the text means.

So before any GPU time I scored 19 scripted policies (no model) under three reward designs:

- A binary verifier reward is exploitable by 7 of my 10 hacks (copy the chunk, cite everything,
  stuff in numbers, answer from the wrong chunk, swap one constant...), all at 0.89 to 1.00. The
  graded v1 does not fix it.
- The final reward (v2) holds every hack I enumerated below 0.50; the highest is 0.44. It needs
  gold labels to do that, and I revised it while looking at the TEST audit, so "closed" is
  in-sample for the attacks I thought of.

Then the training, scored on the 265-task TEST split (EIPs that never appear in training):

- A no-model policy that cites the best-matching line scores 57.4% grounded-correct (0.613 on v2).
  That is the bar a trained model has to clear.
- SFT (Qwen3-0.6B, LoRA r=16, 2 epochs, about 39 minutes on an AMD gfx906 card under ROCm):
  grounded-correct 27.2% -> 98.7% (95% CI 97.0-100.0), verified 40.4% -> 99.6%, mean v2 reward
  0.357 -> 0.985, hallucinated citations 0.8% -> 0.4% (2 of 265 answers -> 1).
- GRPO (300 optimiser steps): zero reward and zero completion length at every step. The merged SFT
  model emitted only the empty no-think block under the GRPO prompt, so nothing was scored and
  nothing was learned. That is a null result about my setup. GRPO was not shown to work or to fail,
  and I claim no benefit from RL.

What this does not show: synthetic, templated questions about 130 EIPs; one seed, one model size,
one machine. Train and test questions share templates, so the SFT model learned a template's answer
pattern; that is not evidence it understands EIPs or would handle a real user's question. The
verifier checks that tokens appear in the cited text, not entailment. The base model abstained on
41.9% of test questions (only 11.3% are unanswerable), so 27.2% is a weak starting point.

Code, audit, leakage check, null-run diagnostic and threats list:
https://github.com/Intermountainh8ter/groundrl

#reinforcementlearning #RLVR #LLM #ethereum #opensource
