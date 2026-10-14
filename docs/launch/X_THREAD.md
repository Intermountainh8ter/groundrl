# X thread

1/
Built a deterministic citation checker as a verifiable reward (RLVR) for small models answering
questions about Ethereum EIPs. Before training anything I tried to break it. Training: not yet run.

2/
The check: every sentence has a [eip-N#CCC] citation, the cited chunk was in context, and every
number / hex / 2**n / ALL_CAPS identifier appears in the chunk it cites.

3/
It is exact and cheap. It is also lexical: it checks token presence, not meaning. So I wrote 19
scripted policies, no model needed, and scored them under three rewards on held-out EIPs.

4/
Binary verifier reward: 7 of my 10 hacks score 0.89-1.00. Copy the chunk, cite everything, stuff
numbers, spam one citation, answer from the wrong chunk, swap one constant, say something vague.

5/
My first richer reward still lost to a hedge: right value, then "(or <wrong value>)". 0.63.
Fix: each extra claim costs a factor of 0.25, plus a consistency term. Highest hack now: 0.44.

6/
It only works with gold labels. Strip them and 7 hacks get through. And copying a one-sentence
chunk is indistinguishable from answering. Both are in the threats list.

7/
Baseline with no model: cite the best-matching line, 0.61 on the reward. That is the bar a
trained model has to clear. The LoRA + GRPO pipeline exists; it has not been run.

8/
Repo, audit table, one-command regeneration, runbook:
github.com/Intermountainh8ter/groundrl
