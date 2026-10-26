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
