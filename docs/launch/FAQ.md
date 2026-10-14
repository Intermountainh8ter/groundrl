# FAQ

**Did you train a model?**
No. The SFT and GRPO pipeline is written, configured and dry-runs without torch, but it has only
been exercised against a fake stack (labelled MOCK). Training results: not yet run. The README says
so, and `scripts/make_report.py` prints "NOT YET RUN" until real result files exist.

**What is real, then?**
The verifier, the three reward versions, the task set (1792 questions split by EIP) and the
reward-hacking audit. The audit scores 19 scripted policies under each reward and regenerates with
`python scripts/reward_audit.py --split test`. A model-free baseline pipeline
(`groundrl baseline-eval`) is real too.

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
The threshold was fixed before any number was produced.

**Is the final reward safe?**
No claim of that. Every hack I enumerated scores below 0.50 under it (highest 0.44), but the list
is mine, the reward was revised after seeing the audit, and a learned policy can find other
attacks. Known residuals are in `docs/THREATS_TO_VALIDITY.md`.

**Why do you need gold labels in the reward?**
Without the gold-based terms 7 of the policies get through. The verifier alone cannot tell a
correct answer from a well-cited wrong one.

**Why is the benchmark easy?**
Most questions can be answered by finding the best-matching line, and a no-model baseline that
does exactly that scores 0.61 on the reward. A trained model only means something if it beats it.

**Can I run it on my GPU?**
There are JSON configs for Qwen3-0.6B and 1.7B at 12 GB and 24 GB. They are untuned defaults and
the memory figures are formulas, not measurements. `docs/RUNBOOK.md` has Windows PowerShell and
Linux steps and rented-GPU notes. If you run it, the result files are what count.

**What would change my mind?**
A trained model that does not beat the extractive baseline; a learned policy that reaches 0.5 on
the final reward without answering; or a flaw in the task generator. Open an issue with the
evidence.
