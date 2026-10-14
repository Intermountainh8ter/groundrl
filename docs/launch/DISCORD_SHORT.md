# Discord short post

**groundrl**: a deterministic citation verifier used as an RL reward, plus a reward-hacking audit.
No model needed to reproduce the audit (`python scripts/reward_audit.py --split test`). 19 scripted
policies; a binary verifier reward is exploitable by 7 of 10 hacks (copy the chunk, cite
everything, stuff numbers...), the final reward holds all of them below 0.50 but needs gold
labels. The LoRA + GRPO training pipeline (Qwen3 0.6B/1.7B, 12/24 GB configs) is written but
**training results: not yet run**. Threats list and runbook in the repo:
github.com/Intermountainh8ter/groundrl
