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

## 2026-10-26T23:55:32Z honest-success-sweep proposal:b49f684be47e79d98f84daa6340e24af610cb513e5a4940ac186a3f9ea12a07c

Hypothesis: on the TEST split with G=8, 20 seeded groups per task and exploit mixing probability p=0.5, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05 at every honest-success probability q in {0.1, 0.25, 0.5, 0.75, 0.9}, while the first_number rung share is at least 0.20 at every q.

Metric: `/exploit_advantage_positive_share/G_8/q_0.9/p_0.5/strict` in `results/57-honest-success-sweep.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung at q=0.9.

Falsifier: a strict-rung share above 0.05 at any q in the sweep, or a first_number-rung share below 0.20 at any q.

## 2026-10-26T23:59:27Z val-split-replication proposal:ba9d2f23fbc101bc1463c62813dbf7d2d3ef436bfd2b1855189515064f4f3390

Hypothesis: on the VAL split with G=8, 20 seeded groups per task, exploit mixing probability 0.5 and honest-success probability 0.5, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05, while the first_number rung share is at least 0.20.

Metric: `/exploit_advantage_positive_share/G_8/q_0.5/p_0.5/strict` in `results/58-val-split-replication.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung; first_number rung at least 0.20.

Falsifier: a strict-rung share above 0.05 on VAL, or a first_number-rung share below 0.20 on VAL.

## 2026-10-27T00:03:04Z seed-replication proposal:0ac0b81e56698f48a80cf60e02d25d8e0af61432aa1b41623a56bb325a37e82b

Hypothesis: on the TEST split with G=8, 20 seeded groups per task, exploit mixing probability 0.5, honest-success probability 0.5 and RNG seed 11, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05, while the first_number rung share is at least 0.20.

Metric: `/exploit_advantage_positive_share/G_8/q_0.5/p_0.5/strict` in `results/59-seed-replication.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung; first_number rung at least 0.20.

Falsifier: a strict-rung share above 0.05 at seed 11, or a first_number-rung share below 0.20 at seed 11.

## 2026-10-27T00:07:03Z train-split-replication proposal:639840144851dedb4adf3e8cd79c4d53a812cc93e86078af354461c5cac4892d

Hypothesis: on the TRAIN split with G=8, 20 seeded groups per task, exploit mixing probability 0.5, honest-success probability 0.5 and seed 7, the strict-rung share of SIMULATED groups with strictly positive exploit mean group-normalised advantage is at most 0.05, while the first_number rung share is at least 0.20.

Metric: `/exploit_advantage_positive_share/G_8/q_0.5/p_0.5/strict` in `results/60-train-split-replication.json`.

Expected: at most 0.05 (range 0.0 to 0.05) on the strict rung; first_number rung at least 0.20.

Falsifier: a strict-rung share above 0.05 on TRAIN, or a first_number-rung share below 0.20 on TRAIN.

## 2026-10-27T00:10:40Z train-seed13-replication proposal:25e47480fae647d35dcc601afcd27dd96b6c8953235b5df91f70e236bb30e882

Hypothesis: on the TRAIN split with seed 13 and 40 groups per task, the SIMULATED positive-advantage share of the strict rung stays exactly 0.0 while the first_number rung stays above 0.5.

Metric: `/exploit_advantage_positive_share/G_8/q_0.5/p_0.5/strict` in `results/61-train-seed13-replication.json`.

Expected: strict positive-advantage share equals 0.0 (range 0.0 to 0.0).

Falsifier: a strict-rung positive-advantage share above 0.0, or a first_number share at or below 0.5.

## 2026-10-27T00:14:25Z extreme-corner-sweep proposal:60589556a111149bc882e39ae458d9a0b1072877b9106df6f5fd558fd4543be3

Hypothesis: on the TEST split at G=32, q=0.1, exploit mixing probability 0.9, 20 seeded groups per task and seed 17, the SIMULATED positive exploit-advantage share of the strict rung is exactly 0.0 and the first_number rung share is at least 0.5.

Metric: `/exploit_advantage_positive_share/G_32/q_0.1/p_0.9/strict` in `results/62-extreme-corner-sweep.json`.

Expected: strict positive-advantage share equals 0.0 (range 0.0 to 0.0).

Falsifier: a strict-rung positive-advantage share above 0.0, or a first_number-rung share below 0.5, at G=32, q=0.1, p=0.9.

## 2026-11-11T09:50:02Z seam-errors proposal:cd526d10c57ac6a82023fc16270206e96a319e770a5b4d5354f1eb64a6f56b71

Hypothesis: at the same marginal false-positive rate (0.10), SIMULATED groups (G=8, TEST split, 20 groups per task, seed 7) where false positives are shared by every sample hitting the same grader seam show a higher share of groups whose advantage sign flips against the strict grader than groups with independent per-sample false positives.

Metric: `/advantage_sign_flip_share/shared_seam` in `results/90-seam-errors.json`.

Expected: shared_seam sign-flip share exceeds the independent share by at least 0.02 absolute (range 0.02 to 1.0).

Falsifier: shared_seam sign-flip share minus independent sign-flip share is below 0.02, or the bootstrap interval of the difference includes zero.

## 2026-11-12T11:23:40Z fuzz-candidates proposal:2f168b7a014e8f90ed3e7102de9483a3cb36895745f4489d063f4977547e11f6

Hypothesis: on the TEST split with seed 7 and 5 mutation classes per task, the SIMULATED exploit-candidate rate (accepted at reward 1.0 by the rung, rejected by the strict rung) of the first_number rung is at least 0.50, while the strict rung is exactly 0.0 by construction and the dup_key rate is strictly lower than the first_number rate.

Metric: `/exploit_candidate_rate/by_rung/first_number` in `results/128-fuzz-candidates.json`.

Expected: first_number exploit-candidate rate at least 0.50 (range 0.5 to 1.0).

Falsifier: a first_number exploit-candidate rate below 0.50, or a dup_key rate not below the first_number rate.

## 2026-11-16T11:46:46Z degeneracy-vs-iid proposal:09b6907ffaa00b8169c41b8e2d160077123474ab2b2488126ad1efb0af333752

Hypothesis: on the TEST split with group size 8, for every ladder rung and every exploit mixing probability, the zero-variance group share predicted from per-task success rates is at least the share predicted from the pooled success rate, so the maximum over cells of (pooled minus per-task) is not positive; the simulated share lies inside the bootstrap interval of the per-task prediction.

Metric: `/degeneracy/pooled_minus_heterogeneous_max_gap` in `results/234-degeneracy-vs-iid.json`.

Expected: metric is at most 0 (range -1.0 to 0.0).

Falsifier: any rung and mixing-probability cell where the pooled prediction exceeds the per-task prediction by more than 1e-9, or a simulated share outside the bootstrap interval of the per-task prediction in most cells.

## 2026-11-16T12:02:08Z verifier-error-asymmetry proposal:726555a62f0e4ba8770ff4783cf27b68069d7979c35fe91ad875daa391f87bfb

Hypothesis: at matched injected error rate, false-positive injection yields a higher exploit share among full-reward rollouts than false-negative injection on the SIMULATED exploit/honest ladder simulator, and the paired bootstrap interval of the difference excludes zero at exploit mixing probability 0.5 under reward v0.

Metric: `/asymmetry/exploit_share_fp_minus_fn` in `results/236-verifier-error-asymmetry.json`.

Expected: metric increases to at least 0.05 (range 0.05 to 1.0).

Falsifier: the 95% bootstrap interval of exploit_share_fp_minus_fn includes zero or lies below zero at the pre-registered cell (v0, mixing probability 0.5, error rate 0.2).

Clarification (written before the run): "reward v0" is mapped to the ladder rung `strict`, the rung that mirrors the binary verifier pass. The pre-registered cell is rung `strict`, exploit mixing probability 0.5, error rate 0.2; all other rungs are reported as secondary cells.
