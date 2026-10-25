# Pre-registration additions

## 2026-10-25T23:26:41Z strictness-ladder proposal:8cc8d7ba8f1bf6494603ad88f0bbe1d8ed7b93934b38581bfe439cf5920b958d

Hypothesis: on the TEST split, a scripted SIMULATED exploiting policy (self-printed output, skipped-item denominator drop, first-number acceptance, duplicate keys) has a mean exploit success rate of at least 0.50 against the loosest ladder rung and at most 0.05 against the strictest rung, and the share of tasks with zero reward spread across the exploit and honest policy pair is higher on the loosest rung than on the strictest.

Metric: `/exploit_success_rate/by_version/strict` in `results/50-strictness-ladder.json`.

Expected: at most 0.05 (range 0.0 to 0.05) for the strictest rung, at least 0.50 for the loosest.

Falsifier: an exploit success rate above 0.05 against the strictest rung, or a loosest-rung rate below 0.50.
