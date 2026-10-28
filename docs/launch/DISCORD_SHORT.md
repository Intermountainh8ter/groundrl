# Discord short post

**groundrl**: a deterministic citation verifier used as an RL reward, a reward-hacking audit of it,
and a first real SFT run. GRPO gave a null result.

- Audit (no model, `python scripts/reward_audit.py --split test`): 19 scripted policies. A binary
  verifier reward is exploitable by 7 of 10 hacks; the final reward holds every hack I enumerated
  below 0.50 (highest 0.44), but needs gold labels and was revised while looking at the TEST audit.
- No-model bar: citing the best-matching line scores 57.4% grounded-correct (0.613 v2) on TEST.
- SFT (Qwen3-0.6B, LoRA, 2 epochs, about 39 min on an AMD gfx906): grounded-correct 27.2% -> 98.7%,
  verified 40.4% -> 99.6%, v2 reward 0.357 -> 0.985 on 265 EIP-disjoint TEST tasks.
- Caveats: templated synthetic questions, one seed, one model size, one machine. Train and test
  share templates, so the model learned a template; the verifier checks tokens, not meaning.
- GRPO: 300 steps, zero reward and zero completion length (the merged SFT model emitted only the
  empty no-think block). A degenerate run, reported as a null result about my setup. GRPO was not
  shown to work or to fail.

Threats list, leakage check, diagnostics: https://github.com/Intermountainh8ter/groundrl
