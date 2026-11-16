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

## 2026-11-16T12:32:10Z audit-allocation proposal:251604506f6baca6fd46fe3f911a19002bd5c650b425f9ab367084ae77169b60

Hypothesis: on the SIMULATED exploit/honest ladder simulator (TEST split, group size 8, 20 groups per task, seed 7, exploit mixing probability 0.5, weak rung first_number, audit fraction 0.1 of full-reward rollouts), the exploit share among rewarded rollouts after audit is lower under targeted audit than under random audit at equal audit budget, and the paired bootstrap interval of (random minus targeted) excludes zero.

Metric: `/audit/exploit_share_random_minus_targeted` in `results/242-audit-allocation.json`.

Expected: metric increases to at least 0.05 (range 0.05 to 1.0).

Falsifier: the 95% bootstrap interval of exploit_share_random_minus_targeted includes zero or lies below zero at the pre-registered cell (rung first_number, mixing probability 0.5, audit fraction 0.1).

## 2026-11-16T13:52:33Z strictness-diversity proposal:da445206a20d83e6258110b324285b0f15c1a910f46c22e185f172d24b62a5e6

Hypothesis: on the SIMULATED exploit/honest ladder simulator (TEST split, group size 8, 20 groups per task, seed 7, exploit mixing probability 0.5), the mean number of distinct output strings among full-reward rollouts per group (groups with at least one full-reward rollout) is higher under the first_number rung than under the strict rung, and the paired bootstrap interval of the difference excludes zero.

Metric: `/strictness/diversity_by_level/p_0.5/first_number_minus_strict` in `results/248-strictness-diversity.json`.

Expected: metric increases to at least 0.5 (range 0.5 to 1.0).

Falsifier: the 95% bootstrap interval of first_number_minus_strict at exploit mixing probability 0.5 includes zero or lies below zero, or the point estimate is below 0.5.

## 2026-11-16T15:44:54Z static-leakiness-probe-resample proposal:10b52efe4fe786c9ad6fd44a76b6aaef7111eb3ddec21f53b9141814dc1d1098

Hypothesis: on the TEST split with seed 7, the Spearman rank correlation (average ranks for ties) across the 20 (rung, mutation class) cells between the static probe-battery acceptance share and the fuzz exploit-candidate rate is at least 0.8, and the lower bound of its 95% joint probe-and-task bootstrap interval is above 0.5.

Metric: `/rank_agreement/spearman_audit_vs_exploit_candidate_rate` in `results/265-static-leakiness-probe-resample.json`.

Expected: metric is at least 0.80 (range 0.8 to 1.0).

Falsifier: the Spearman correlation is below 0.80, or the lower bound of the joint probe-and-task 95% bootstrap interval is at or below 0.5, at the pre-registered 20-cell comparison.

## 2026-11-16T16:59:58Z reward-version-advantage proposal:62193a9d11a8ecc6c55e49fade153a682e4bf6e915aeeb190709d625e24ee2dd

Hypothesis: on the TEST split with G=8, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, each group mixing the scripted oracle policy with one of the seven required audit hack policies (copy_chunk, cite_everything, stuff_numbers, citation_spam, wrong_chunk, one_wrong_constant, paraphrase_no_tokens), the SIMULATED share of groups whose hack rollouts have strictly positive mean group-normalised advantage, averaged over those seven hacks, is at most 0.05 under reward v2 and at least 0.30 under reward v0.

Metric: `/exploit_advantage_by_reward/v2/mean_positive_share` in `results/277-reward-version-advantage.json`.

Expected: metric at most 0.05 under v2 (range 0.0 to 0.05); v0 share at least 0.30.

Falsifier: the v2 mean positive-advantage share across the seven hacks exceeds 0.05, or the v0 share is below 0.30, at the pre-registered cell.

## 2026-11-16T17:24:31Z degenerate-group-by-reward proposal:410b61e356a4eb30708e8a2901cf7718059a315318083b7a5b82d8823952fd92

Hypothesis: on the TEST split with G=8, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups with all-identical rewards (zero group-normalised advantage) is at most 0.10 under reward v2 and strictly higher under reward v0 than under v2.

Metric: `/degenerate_group_fraction_by_reward/v2/mean_share` in `results/282-degenerate-group-by-reward.json`.

Expected: metric at most 0.10 under v2 (range 0.0 to 0.10); v0 share strictly greater than v2 share.

Falsifier: the v2 mean zero-advantage group share exceeds 0.10, or the v0 mean share is not strictly greater than the v2 share, at the pre-registered cell.

## 2026-11-16T18:43:41Z degenerate-group-size2 proposal:bb035540f35591fba56dd07e244c6f231a20417d4519f33716a27a9a869479ac

Hypothesis: on the TEST split with G=2, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups with all-identical rewards (zero group-normalised advantage) under reward v2 is at least 0.40, i.e. more than ten times the G=8 value in `results/282-degenerate-group-by-reward.json`, and v2 stays below v1 and v0.

Metric: `/degenerate_group_fraction_by_reward/v2/mean_share` in `results/285-degenerate-group-size2.json`.

Expected: metric increases to at least 0.40 (range 0.40 to 0.70); v2 below v1 below v0.

Falsifier: the v2 mean zero-advantage group share at G=2 is below 0.40, or v2 is not strictly below both v1 and v0.

## 2026-11-16T19:07:40Z degenerate-group-size4 proposal:78adf54689a273435e9abeef87ace19a954b2843f64977870bf1ace24673e90c

Hypothesis: on the TEST split with G=4, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups with all-identical rewards (zero group-normalised advantage) under reward v2 lies strictly between its G=8 value in `results/282-degenerate-group-by-reward.json` and its G=2 value in `results/285-degenerate-group-size2.json`, is at least 0.05 and at most 0.30, and v2 is strictly below v1 and v0.

Metric: `/degenerate_group_fraction_by_reward/v2/mean_share` in `results/289-degenerate-group-size4.json`.

Expected: metric between 0.05 and 0.30 (above the G=8 value, below the G=2 value).

Falsifier: the v2 mean zero-advantage group share at G=4 is below 0.05 or above 0.30, or is not strictly between the G=8 and G=2 values, or v2 is not strictly below both v1 and v0.

## 2026-11-16T19:14:17Z sign-flip-v0-v2 proposal:90d7b8fabeb6e7fa36cce69d7eb6de26cd9da062e677dbf44e09588cf4f8e2a8

Hypothesis: on the TEST split with G=8, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups whose mean hack-rollout group-normalised advantage has a different sign (negative, zero or positive; tolerance 1e-9) under reward v0 than under reward v2 is at least 0.70. Denominator: groups containing at least one hack rollout. Strict positive/negative reversals are reported separately.

Metric: `/sign_flip_share/v0_vs_v2/mean_share` in `results/290-sign-flip-v0-v2.json`.

Expected: metric at least 0.70 (range 0.70 to 1.0).

Falsifier: the mean v0-vs-v2 sign-change share across the seven hacks is below 0.70 at the pre-registered cell.

## 2026-11-16T19:20:12Z sign-flip-group-size4 proposal:c6440a82960142fe8561d38ede7af3f1d23387654e7606ff6ccc342debe2c850

Hypothesis: on the TEST split with G=4, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups containing a hack rollout whose mean hack-rollout group-normalised advantage has a different sign (negative, zero or positive; tolerance 1e-9) under reward v0 than under reward v2 is at least 0.70.

Metric: `/sign_flip_share/v0_vs_v2/mean_share` in `results/291-sign-flip-group-size4.json`.

Expected: metric stays at or above 0.70 at G=4 (range 0.70 to 1.0).

Falsifier: the mean v0-vs-v2 sign-change share across the seven hacks is below 0.70 at G=4.

## 2026-11-16T19:32:34Z sign-flip-group-size2 proposal:e16236032d55036af4e2ccd29c2d2040add2226dfc9a06b4b8e2dba667f37d5c

Hypothesis: on the TEST split with G=2, 20 seeded groups per task, hack mixing probability 0.5 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups containing a hack rollout whose mean hack-rollout group-normalised advantage has a different sign (negative, zero or positive; tolerance 1e-9) under reward v0 than under reward v2 is strictly below its G=4 value in `results/291-sign-flip-group-size4.json` (0.8958503586543898) and at least 0.50.

Metric: `/sign_flip_share/v0_vs_v2/mean_share` in `results/293-sign-flip-group-size2.json`.

Expected: metric decreases below 0.8958503586543898 but stays at least 0.50.

Falsifier: the mean v0-vs-v2 sign-change share at G=2 is at or above the G=4 value 0.8958503586543898, or below 0.50.

## 2026-11-16T19:46:50Z sign-flip-prob025 proposal:6ce1ca96c709ff0e02f8c967a9f83c9f7f09643290e49c4e57a5af7a022ca7c9

Hypothesis: on the TEST split with G=8, 20 seeded groups per task, hack mixing probability 0.25 and seed 7, averaged over the seven required audit hack policies, the SIMULATED share of groups containing a hack rollout whose mean hack-rollout group-normalised advantage has a different sign (negative, zero or positive; tolerance 1e-9) under reward v0 than under reward v2 is at least 0.85.

Metric: `/sign_flip_share/v0_vs_v2/mean_share` in `results/296-sign-flip-prob025.json`.

Expected: metric stays at or above 0.85 at mixing probability 0.25 (range 0.85 to 1.0).

Falsifier: the mean v0-vs-v2 sign-change share across the seven hacks is below 0.85 at mixing probability 0.25, G=8.

## 2026-11-16T20:01:00Z audit-allocation-frac05 proposal:081e40472b7bfcd2ae86a065ebf8f51806bee3af07c74c50a5a2ed1333cee042

Hypothesis: on the SIMULATED exploit/honest ladder simulator (TEST split, group size 8, 20 groups per task, seed 7, exploit mixing probability 0.5, weak rung first_number), at audit fraction 0.5 of full-reward rollouts the exploit share among rewarded rollouts after audit under random audit minus targeted audit stays below 0.05, because full-reward rollouts tie under the weak verifier so targeting by weak reward has no signal to use.

Metric: `/audit/exploit_share_random_minus_targeted` in `results/298-audit-allocation-frac05.json`.

Expected: metric stays below 0.05 (near 0.0, as at audit fraction 0.1; range 0.0 to 0.05).

Falsifier: exploit_share_random_minus_targeted at the pre-registered cell is at least 0.05 at audit fraction 0.5.
