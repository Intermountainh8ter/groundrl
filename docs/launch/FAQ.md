# FAQ

**Did you train a model?**
Yes, once with SFT and once with GRPO. SFT (Qwen3-0.6B, LoRA r=16, 2 epochs, seed 0, about 39
minutes on an AMD gfx906 card under ROCm) produced the results below. The GRPO run (300 optimiser
steps on the merged SFT model) was degenerate: zero reward and zero completion length at every
logged step, so it taught nothing and its row in the README reads DEGENERATE. Result files are in
`results/groundrl-sft/` and `results/groundrl-grpo/`; `scripts/make_report.py` renders them into
`results/REPORT.md`.

**What did SFT score?**
On the 265-task TEST split (EIPs that never appear in training), against the untouched base
Qwen3-0.6B with the same prompt: grounded-correct 27.2% -> 98.7% (95% CI 97.0-100.0), verified
40.4% -> 99.6%, hallucinated citations 0.8% -> 0.4% (2 of 265 answers -> 1, not a meaningful
difference), mean reward v2 0.357 -> 0.985. The no-model `extractive` policy scores 57.4%
grounded-correct and 0.613 on v2 on the same split; that is the bar a trained model has to clear.

**Does that mean the model understands EIPs?**
No. The questions are synthetic templates about 130 EIPs, and train and test questions are drawn
from the same templates. The split removes shared EIPs and shared facts, not the question form, so
the SFT model learned a template's answer pattern that the test questions share. It is one seed,
one model size and one machine. The verifier checks that tokens appear in the cited text, not that
the text entails the answer. Nothing here estimates how the model would do on questions a real user
writes. The base model is also a weak comparison: it abstained on 41.9% of test questions, though
only 11.3% are unanswerable, and I ran no other baseline (few-shot, a larger model).

**What happened with GRPO?**
All 300 optimiser steps reported zero reward, zero completion length and zero gradient norm. Under
the GRPO prompt rendering the merged SFT model emitted only the empty no-think block and stopped,
so the trainer had nothing to score. That is a null result about my setup, not a finding about
GRPO: GRPO with reward v2 was not shown to work or to fail, and I claim no benefit from RL.
`results/groundrl-grpo/DIAGNOSTIC.md` has the trace and `docs/METHODOLOGY.md` section 6a the
hypotheses for a new run.

**Why not tune the settings and rerun?**
The settings were fixed before the run, and changing them after seeing a result is the thing a
pre-registration is meant to prevent. A fix is a new run with its own pre-registered config, after
one of the hypotheses in section 6a is confirmed as the cause. Until then there is no GRPO number.

**Is the test set clean?**
There is zero EIP overlap and zero fact-id overlap between train and test
(`results/groundrl-sft/LEAKAGE_CHECK.md`, re-derived from `data/tasks`). That does not remove
conceptual leakage between related EIPs, possible exposure of Qwen3 to the EIPs in pretraining, or
the shared question templates described above.

**What is real?**
The verifier, the three reward versions, the task set (1792 questions split by EIP), the
reward-hacking audit, the model-free baselines (`groundrl baseline-eval`), the SFT run and its
evaluation, and the degenerate GRPO run. The audit scores 19 scripted policies under each reward
and regenerates with `python scripts/reward_audit.py --split test`.

**Why a deterministic verifier instead of an LLM judge?**
It is exact, free, reproducible and cannot be talked into agreeing with the policy. The price is
that it checks token presence, not meaning, which is why the audit exists.

**How is this different from `groundrails`?**
As far as I read its package description, `groundrails` is a deterministic claim-verification
guardrail for agentic RAG. This repository uses a similar kind of check as a training reward and
studies how that reward can be gamed. I found no prior work on citation-grounding rewards for RL,
from a limited search.

**What does "exploitable" mean here?**
A policy that does not answer the question gets a mean reward of at least 0.50 on the test split.
The threshold was fixed before any number was produced. Under the binary verifier reward, 7 of my
10 required hacks are exploitable (0.89 to 1.00); the final reward v2 leaves none at 0.50 or above.

**Is the final reward safe?**
No claim of that. Every hack I enumerated scores below 0.50 under it (highest 0.44), but the list
is mine, the reward was revised after seeing the audit, and a learned policy can find other
attacks. No policy has been optimised against it, because the GRPO run was degenerate, so how it
holds up against a learned policy is untested. Known residuals are in
`docs/THREATS_TO_VALIDITY.md`.

**Why do you need gold labels in the reward?**
Without the gold-based terms 7 of the policies get through. The verifier alone cannot tell a
correct answer from a well-cited wrong one.

**Why is the benchmark easy?**
Most questions can be answered by finding the best-matching line, and a no-model baseline that
does exactly that scores 0.61 on the reward. A trained model only means something if it clears
that bar. The SFT model does (0.985), but on questions that share the training templates, so the
margin is on-template.

**Can I run it on my GPU?**
There are JSON configs for Qwen3-0.6B and 1.7B at 12 GB and 24 GB, and the AMD gfx906 config the
recorded runs used (`docs/ROCM_GFX906.md` lists the pitfalls). Only the gfx906 config has been run;
the others are untuned defaults and their memory figures are formulas, not measurements.
`docs/RUNBOOK.md` has Windows PowerShell and Linux steps and rented-GPU notes. If you run it, the
result files are what count.

**What would change my mind?**
A rerun with other seeds where the SFT score falls toward the extractive baseline; an SFT model that
does well on templated questions and fails on questions that do not follow the templates; a
learned policy that reaches 0.5 on the final reward without answering; or a flaw in the task
generator. Open an issue with the evidence.

**Was AI used to build this?** Yes. An AI coding assistant wrote most of the code, tests and prose in this repository (the commits are co-authored in the git history). I directed the work and ran the real-model benchmarks on my own hardware. The README tables are generated by scripts from committed result files, and tests recompute most of the figures quoted in the README and the launch posts; a few hand-written figures are not machine-checked. Read the code and the methodology before trusting any conclusion.
