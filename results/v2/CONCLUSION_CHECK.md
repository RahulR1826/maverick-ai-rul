# Does the original conclusion still hold?

**Short answer: partly.** FedAvg still matches or beats the other methods on FD001 under the conditions tested,
but the original evidence overstated the size of that advantage, and three of the seven original findings do not
survive repeated, budget-matched experiments.

Comparison of the original findings (`results/FINAL_REPORT.md` §13–14: 1–3 seeds, centralized trained with early
stopping, FL with a fixed budget, single-seed sweeps) with the v2 protocol (5 matched seeds, identical training
budget and checkpoint-selection rule for every method). New values are test RMSE, mean ± SD over 5 seeds, in cycles.
"Selected" = checkpoint chosen on the 10 validation engines; "final" = model after the full budget. All new numbers
are in `tables/summary.csv`; per-seed values are in `tables/per_seed.csv`.

| # | Original finding | Original value(s) | New result (5 seeds) | Verdict |
|---|---|---|---|---|
| 1 | FL-IID beats centralized by ≈3 cycles | FL 12.95 vs Cent 16.01 (seed 42 only); gap 3.06. The original 3-seed means were 13.21 vs 14.61 (gap 1.40) but were not the headline | FedAvg-IID 13.24 ± 0.32 vs budget-matched centralized 14.46 ± 0.44 (selected); gap **+1.22 ± 0.56**, FedAvg lower in **5/5** seeds. Final checkpoint: 12.88 vs 14.82 | **Direction holds; magnitude does not.** The gap is ≈60% smaller. It comes from the centralized model over-fitting under shared, untuned settings, not from decentralisation. Optimizer reset alone (K=1 ablation, 13.94 ± 0.89) does not reproduce it. |
| 2 | More communication rounds help; R=30 best | R5 14.14 → R30 12.50 (best), R40 13.20 | Final: R5 14.83 ± 0.94, R10 13.78 ± 0.26, **R20 12.88 ± 0.30**, R30 13.33 ± 0.22 | **Partly holds.** Rounds help up to R=20, then plateau or slightly degrade. "R=30 is best" **does not hold**: it was a single-seed result. |
| 3 | E=1 much worse; E≈5 best | E1 19.55, E3 13.38, E5 12.65, E10 13.30 | Final: E1 17.42 ± 1.51, E3 13.11 ± 0.24, **E5 12.88 ± 0.30**, E10 13.67 ± 0.39 | **Holds.** E=1 is clearly worse. E=5 is lowest on average, but E=3 is within one SD. |
| 4 | More clients improve accuracy (K2 → K10) | K2 15.12, K5 13.61, K10 13.09 | IID-window (final): K2 13.62 ± 0.40, **K5 12.88 ± 0.30**, K10 13.53 ± 0.32. Whole-engine (selected): K2 13.65, K5 13.87, K10 14.34 | **Does not hold.** Not monotone under IID-window. Under whole-engine ownership error *rises* with K. Total data is fixed, so more clients means less data each. |
| 5 | Non-IID (Dirichlet α) has little effect | IID 13.45, α1.0 13.84, α0.5 13.38, α0.1 13.58 (mostly 1 seed) | Selected: IID 13.24 ± 0.32, α1.0 13.35 ± 0.26, α0.5 13.47 ± 0.32, **α0.1 14.06 ± 0.46** | **Partly holds.** α ≥ 0.5 is within ≈1 SD of IID. α=0.1 is worse (+0.82), but it also has extreme quantity skew (479–8,555 windows per client), so this cannot be called a pure label-heterogeneity effect. |
| 6 | Engine-level ownership is worse than IID | 14.70 (seed 42), 3-seed mean ≈14.46 | Whole-engine 13.87 ± 0.19 vs IID-window 13.24 ± 0.32; gap +0.63 ± 0.42, worse in **5/5** seeds | **Holds, but smaller and differently explained.** Client RUL distributions are nearly identical, so the gap reflects engine ownership, cross-client duplicate windows and volume, not label heterogeneity. |
| 7 | Primary FL communication = 164.55 MB | Analytical 2·R·4·P·K | Measured serialized payload: 864,260 B/model, **172.85 MB = 164.84 MiB** total (K=5, R=20) | **Holds with a unit correction.** The original number was MiB and excluded .npy headers. It's still an estimate: no network was measured, protocol overhead excluded. |

## New findings the original paper does not have

* **Local-only vs FedAvg** (new baseline). Whole-engine local models (18 engines per airline) reach 16.28 ± 0.50 RMSE
  and NASA 652 ± 80. FedAvg on the same partition reaches 13.87 ± 0.19 and NASA 349 ± 26, lower in 5/5 seeds.
  **This is the strongest practical argument for federation in the paper** and should lead the results section.
* **Centralizing the raw data would be ≈130× cheaper in bytes** than FedAvg's model traffic at R=20 (≈1.3 MB
  vs 172.9 MB). FedAvg's case on FD001 is data ownership, not bandwidth.
* **All methods over-predict RUL on average** (+3.3 to +4.8 cycles, 58–69% late), which drives the NASA score.

## What should change in the manuscript

1. Replace every single-seed number with **mean ± SD over 5 seeds**, and state the seeds and that they are matched.
2. Reframe the central claim: *"Under a matched training budget and identical untuned hyperparameters, FedAvg matched
   or exceeded pooled centralized training on FD001 (RMSE 13.24 ± 0.32 vs 14.46 ± 0.44); an ablation indicates this
   reflects the regularising effect of model averaging rather than a benefit of decentralisation."* Do **not** claim
   that FL is inherently more accurate than centralized learning.
3. Lead with **FedAvg vs local-only under whole-engine ownership** (−2.41 RMSE, NASA 652 → 349).
4. Rewrite the rounds and clients subsections: R=20 (not R=30) is best, and accuracy does **not** improve
   monotonically with K. State that total training data is fixed as K grows.
5. Describe α=0.1 and whole-engine results with their confounds (quantity skew; engine ownership and duplicate windows).
6. Report communication as **172.85 MB (serialized-payload estimate; no protocol overhead; simulated)**, with MB/MiB stated.
7. Remove or qualify any privacy claim: FedAvg keeps raw data local but provides **no formal privacy guarantee**.
8. Make no significance claims. With 5 seeds the smallest attainable two-sided paired p-value is 0.0625; report
   per-seed consistency (e.g. "lower in 5/5 seeds") instead.
