# Pre-registration additions

## 2026-10-25T23:26:41Z strictness-ladder proposal:8cc8d7ba8f1bf6494603ad88f0bbe1d8ed7b93934b38581bfe439cf5920b958d

Hypothesis: on the TEST split, a scripted SIMULATED exploiting policy (self-printed output, skipped-item denominator drop, first-number acceptance, duplicate keys) has a mean exploit success rate of at least 0.50 against the loosest ladder rung and at most 0.05 against the strictest rung, and the share of tasks with zero reward spread across the exploit and honest policy pair is higher on the loosest rung than on the strictest.

Metric: `/exploit_success_rate/by_version/strict` in `results/50-strictness-ladder.json`.

Expected: at most 0.05 (range 0.0 to 0.05) for the strictest rung, at least 0.50 for the loosest.

Falsifier: an exploit success rate above 0.05 against the strictest rung, or a loosest-rung rate below 0.50.

## 2026-10-26T23:33:10Z group-degeneracy proposal:7afc3031cf1bc51ea8789c69e60a35502b3ae07aab829fa95cde7a204cb1f8af

Hypothesis: on the TEST split with group size 8, 20 seeded groups per task and exploit mixing probability 0.5, the share of groups with zero reward variance is at least 0.50 on the loosest rung (first_number) and at most 0.05 on the strict rung, so a stricter verifier restores GRPO advantage signal in a mixed honest/exploit group. SIMULATED sampler, no model.

Metric: `/zero_variance_group_share/p_0.5/strict` in `results/51-group-degeneracy.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung, at least 0.50 on the loosest.

Falsifier: a strict-rung zero-variance group share above 0.05 at p=0.5, or a first_number rung share below 0.50.

## 2026-10-26T23:36:58Z exploit-advantage proposal:bc160952e085e761614fd40533547e74cc89e1fa17c649303e825b598b695b87

Hypothesis: on the TEST split with group size 8, 20 seeded groups per task and exploit mixing probability 0.5, the share of SIMULATED groups whose exploit rollouts have a strictly positive mean group-normalised advantage is at most 0.05 on the strict rung and at least 0.20 on at least one of the dup_key and skip_denominator rungs. Groups with zero reward variance or with no exploit rollout count as not positive.

Metric: `/exploit_advantage_positive_share/p_0.5/strict` in `results/52-exploit-advantage.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung.

Falsifier: a strict-rung exploit-positive-advantage share above 0.05 at p=0.5, or both dup_key and skip_denominator below 0.20.

## 2026-10-26T23:40:52Z noisy-honest-advantage proposal:bb2cc27c9e417d42c4967f16051128fd77b3491b72cdfa4a975175c96902dbb2

Hypothesis: on the TEST split with group size 8, 20 seeded groups per task, exploit mixing probability 0.5 and honest-success probability q=0.5, the share of SIMULATED groups whose exploit rollouts have strictly positive mean group-normalised advantage is at most 0.05 on the strict rung and at least 0.20 on at least one of first_number, dup_key and skip_denominator. Each honest rollout is the gold answer with probability q, else a corrupted gold answer (first item replaced, pass line dropped; abstention replaced by a non-abstaining answer for unanswerable tasks).

Metric: `/exploit_advantage_positive_share/q_0.5/p_0.5/strict` in `results/53-noisy-honest-advantage.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung; at least one looser rung at least 0.20.

Falsifier: a strict-rung share above 0.05 at q=0.5, p=0.5, or all three looser rungs below 0.20.

## 2026-10-26T23:48:21Z group-size-sweep proposal:650e73bb17e7368541429bdedd22f548917e6c4ef608b2185aaac81aa0bc6e25

Hypothesis: on the TEST split with 20 seeded groups per task, exploit mixing probability 0.5 and honest-success probability 0.5, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05 at every group size G in {2, 4, 8, 16}, while the first_number rung share is at least 0.20 at every G.

Metric: `/exploit_advantage_positive_share/G_16/q_0.5/p_0.5/strict` in `results/55-group-size-sweep.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung at G=16.

Falsifier: a strict-rung share above 0.05 at any G in {2, 4, 8, 16}, or a first_number-rung share below 0.20 at any G.

## 2026-10-26T23:52:05Z mixing-prob-sweep proposal:e8cc328283ca23e3b2c30348c267197b99ac165a2ca2fd5b7de1817af43af367

Hypothesis: on the TEST split with G=8, 20 seeded groups per task and honest-success probability q=0.5, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05 at every exploit mixing probability p in {0.1, 0.25, 0.5, 0.75, 0.9}, while the first_number rung share is at least 0.20 at every p.

Metric: `/exploit_advantage_positive_share/G_8/q_0.5/p_0.9/strict` in `results/56-mixing-prob-sweep.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung at p=0.9.

Falsifier: a strict-rung share above 0.05 at any p in the sweep, or a first_number-rung share below 0.20 at any p.
