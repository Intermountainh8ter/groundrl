# Glossary

Every abbreviation and piece of jargon used in this repository, in plain words.

## Reinforcement learning

**abstention** — The model declining to answer because the documents do not
contain the answer. Rewarded here, because a confident wrong answer is worse
than "I don't know".

**GRPO — Group Relative Policy Optimization** — The reinforcement-learning
method used here. It generates several answers to the same question, scores
them, and nudges the model toward the ones that scored above the group's
average. No separate value model is needed.

**KL penalty (beta)** — A brake that keeps the model being trained from drifting
too far from the model it started as. Without it, reward-chasing can destroy
general ability.

**LoRA — Low-Rank Adaptation** — A way to fine-tune a large model by training a
small number of extra parameters instead of all of them. Cheap, and the result
can be merged back into the original weights.

**policy** — In reinforcement learning, the thing being trained: here, the model
that writes answers. Also used in this repo for a *scripted* policy — a
hand-written fake answerer used to test the reward.

**reward hacking** — Scoring well without doing the real task, by exploiting a
loophole in how scoring works. The thing this repo is built to catch.

**RL — Reinforcement Learning** — Training by scoring the model's own output and
reinforcing what scored well, rather than copying fixed correct answers.

**RLVR — Reinforcement Learning from Verifiable Rewards** — RL where the score
comes from a checkable fact (does the citation exist? is the number right?)
rather than from human preference or another model's opinion.

**rollout / generation** — One answer produced by the model during training.

**SFT — Supervised Fine-Tuning** — The first training stage: show the model
correct answers and have it imitate them. Done before GRPO here.

**TRL — Transformer Reinforcement Learning** — The HuggingFace library that
implements SFT and GRPO. `trl_backend.py` wraps it.

**v0 / v1 / v2** — The three successive designs of this repo's reward function.
v2 is the one that ships; v0 and v1 are kept because the audit shows both were
exploitable, and v1 was *worse* than v0.

## This repo's reward terms

**abstention term** — Credit for correctly declining an unanswerable question.

**citation term** — Credit for pointing at the chunk that actually supports the
claim.

**consistency term** — Penalty for an answer that contradicts itself.

**correctness term** — Credit for the answer matching the gold answer. The term
that uses the gold label, and the one most cheats need removed to win.

**gold answer / gold label** — The known-correct answer, written in advance.

**gold-free** — A scoring variant using no gold labels at all. Reported because
five cheats beat it, which is an honest limit on the method.

**length term / redundancy term / relevance term** — Penalties for padding, for
repeating the same content, and for citing material that is not on topic.

## Evaluation

**audit** — Running hand-written cheating policies against the reward to see
which ones win, *before* training anything.

**bootstrap** — Estimating uncertainty by resampling the data many times.

**exploitable** — A policy whose mean reward reaches 0.50, a threshold fixed
before any policy was run.

**fires on** — The share of tasks where a policy's output differs from the gold
answer, i.e. where its trick actually applies.

**format drift** — The model stopping following the required output format.

**split: train / val / test** — Three separate sets of tasks. Train is for
learning, val for checking during development, test is scored once at the end.

## The subject matter

**EIP — Ethereum Improvement Proposal** — The numbered documents that specify
changes to Ethereum. The corpus this repo answers questions about.

**ERC — Ethereum Request for Comment** — A category of EIP covering application
standards, such as token formats.

**opcode** — A single low-level instruction in the Ethereum virtual machine.

**RAG — Retrieval-Augmented Generation** — Answering a question by first
retrieving relevant documents and then writing from them.

## Files, formats and systems

**CC0** — A public-domain dedication; the licence of the corpus data.

**CLI — Command-Line Interface** — The terminal commands this tool provides.

**CPU / GPU** — The general-purpose processor, and the parallel processor that
trains and runs models.

**gfx906** — The AMD GPU architecture (Vega 20) this repo has specific setup
instructions for. See `docs/ROCM_GFX906.md`.

**GB / GiB** — Units of storage. A GiB is 1024³ bytes, slightly larger than a GB.

**JSON** — A structured record format.

**MIT** — The permissive open-source licence this repo's code uses.

**MOCK** — A run using a fake model stack, to test the plumbing. Mock output is
labelled and must never be reported as a result.

**NOT YET RUN** — The marker this repo puts on any headline number that has not
actually been measured, instead of estimating it.

**ROCm / HIP / CUDA** — AMD's GPU software stack; AMD's programming interface;
and NVIDIA's equivalent.

**VRAM** — Memory on the graphics card. The limit on how large a model you can
train.
