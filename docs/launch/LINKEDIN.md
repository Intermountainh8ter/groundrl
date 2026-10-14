# LinkedIn post

I built the reward function for an RL experiment I have not run yet, and I stress-tested it first.

The idea: use a deterministic citation checker as a verifiable reward (RLVR) to train a small model
(Qwen3 0.6B / 1.7B class, LoRA + GRPO) to answer questions about Ethereum EIPs with grounded
citations. The checker is simple and exact. Every sentence needs a [eip-N#CCC] citation, the cited
chunk must have been in the context, and every number, hex value, 2**n and ALL_CAPS identifier must
appear in the chunk it cites.

Simple and exact also means gameable. So before any GPU time, I wrote 19 scripted "policies" that
need no model and scored them under three reward designs on a held-out split:

- A binary verifier reward (v0) is exploitable by 7 of the 10 hacks I tried: copy the whole chunk,
  cite everything, stuff in every number, repeat a citation, answer from the wrong chunk, swap one
  constant for another number from the same chunk, or write something vague with a citation. All
  score 0.89 to 1.00.
- My first attempt at a richer reward still lost to two "hedging" answers (the right value, then
  "or <wrong value>"), at 0.63.
- The revised reward holds every hack I enumerated below 0.50. The highest is 0.44. It needs gold
  labels to do that: without them, 7 policies get through.

The table, the revision log, and the one-command way to regenerate it are in the repo. So are the
holes that remain: copying a one-sentence chunk is indistinguishable from answering, and a trained
policy may find attacks I did not think of.

What I have not done: train anything. The SFT and GRPO pipeline is written, configured for 12 and
24 GB cards, dry-runs without torch, and has only been exercised against a fake stack. Training
results: not yet run. A trivial "find the best-matching line and cite it" baseline scores 0.61 on
my reward, which is the bar a trained model has to clear.

Code, audit, and the threats-to-validity list: github.com/Intermountainh8ter/groundrl

#reinforcementlearning #RLVR #LLM #ethereum #opensource
