## 10. Interpretation (written after inspecting the results above)

All statements below refer to numbers in sections 2–9 of this report or in `tables/*.csv`.
"Lower" means lower test error. With 5 seeds no difference is declared statistically
significant (see section 3); we describe how large and how consistent the effects are across matched seeds.

### 10.1 Centralized vs FedAvg vs local-only

* **FedAvg-IID had lower test error than budget-matched centralized training in all 5 matched seeds**
  (RMSE 13.24 ± 0.32 vs 14.46 ± 0.44 at the validation-selected checkpoint; per-seed gap +0.29 to +1.73 cycles,
  mean +1.22). The same ordering holds for MAE and the NASA score, and at the final checkpoint.
  The original report's gap (3.06 cycles: 12.95 vs 16.01) came mostly from comparing a single favourable FL seed with
  a single unfavourable centralized seed. Under matched seeds the gap is **about 60% smaller**.
* **Why FedAvg can beat centralized training here.** Federation itself adds no information, so the advantage has to
  come from the optimisation procedure. The convergence plot (fig. 2) shows the centralized model reaching its best
  validation RMSE around epoch 45 of 100 and then **overfitting** (validation RMSE rises from ≈13 to ≈15). FedAvg's
  validation RMSE keeps improving slowly through round 20. Two plausible regularising mechanisms are (i) model
  averaging across clients and (ii) the per-round reset of each client's Adam state.
  **Ablation (`ablation_fedavg_k1`, 5 seeds):** FedAvg with a single client holding all data is exactly centralized
  training with the Adam state reset every 5 epochs and the same selection rule. It reaches 13.94 ± 0.89 (selected)
  and 15.96 ± 1.62 (final), between centralized (14.46 / 14.82) and FedAvg-IID (13.24 / 12.88). It is lower than
  FedAvg-IID in only 2 of 5 seeds, selects early (round 7.2 on average) and overfits afterwards. **The optimizer
  reset alone therefore does not reproduce FedAvg's advantage.** The evidence is consistent with averaging five
  clients' updates acting as a regulariser: each client takes ≈65 optimizer steps per round instead of ≈310 on the
  pooled data, and their average moves the global model more conservatively. The centralized baseline is effectively
  over-trained at 100 epochs with these settings. This is a property of the optimisation set-up, not a benefit of
  data decentralisation, and the manuscript should say so.
* **Hyperparameters were not tuned for either method.** All methods use the original repository's settings
  (Adam 1e-3, batch 256, dropout 0.2, no LR schedule). A centralized model with tuned regularisation or a tuned epoch
  budget may close the gap. The defensible claim is "under identical, untuned hyperparameters and a matched
  training budget", not "FedAvg is more accurate than centralized learning".
* **The original early-stopping protocol does not explain the original gap either.** `centralized_legacy_es`
  (original early stopping + ReduceLROnPlateau) gives 14.55 ± 0.48, practically the same as the
  budget-matched centralized run (14.46 ± 0.44; difference well within one SD). Changing the centralized protocol therefore removes a
  confound in the paper without changing the centralized number much.
* **Local-only training is clearly worse than FedAvg**: IID-window local models 14.58 ± 0.17 (+1.34 vs FedAvg-IID,
  5/5 seeds). Under whole-engine ownership local models (18 engines each) reach 16.28 ± 0.50 vs 13.87 ± 0.19 for
  FedAvg on the same partition (−2.41 cycles), and the NASA score drops from 652 to 349. **This is the strongest and
  most practically relevant result:** an airline with 18 engines gains substantially from federating with four others.
  IID-window local-only is less informative: each local shard contains fragments of all 90 engines, which a real
  airline would not have.

### 10.2 Partition protocols and heterogeneity

* **Whole-engine ownership costs 0.63 ± 0.42 RMSE relative to IID-window FedAvg (5/5 seeds)**
  (13.87 vs 13.24). This should **not** be described as an effect of statistical heterogeneity. Client RUL
  distributions are almost identical under both partitions (client mean RUL 73.5–82.8 vs 78.1–81.8 cycles; fig. 9),
  but the partitions differ in (a) engine ownership (each IID-window client sees fragments of all 90 engines, each
  whole-engine client sees 18 complete trajectories), (b) cross-client near-duplicate windows (overlapping windows of
  one engine split across clients), and (c) volume (3,126 windows per client vs 2,751–3,352).
* **Dirichlet label skew degrades FedAvg only at α = 0.1**: 13.35 ± 0.26 (α=1.0), 13.47 ± 0.32 (α=0.5),
  14.06 ± 0.46 (α=0.1) vs 13.24 ± 0.32 (IID). The α=1.0 and α=0.5 differences are within one SD. At α=0.1 the
  partition is extremely skewed in **both** label and quantity (client means 20.8–112.8 cycles; client sizes 479–8,555
  windows; one redraw needed for one seed to satisfy the 256-window minimum). The degradation therefore cannot be
  attributed to label skew alone.
* Validation RMSE ranks the conditions the same way as test RMSE (IID 12.36 < α1.0 12.45 < α0.5 12.69 = whole-engine
  12.87 < α0.1 13.22), which supports the ordering.

### 10.3 Error characteristics

* All methods **over-predict RUL on average** (mean error +3.3 to +4.8 cycles; 58–69% of test engines predicted late).
  Late predictions are the costly direction under the NASA score, which is why the score separates methods more than
  RMSE does. FedAvg-IID has the smallest bias (+3.3) and the smallest 90th-percentile absolute error (21.8 cycles).
* The NASA score is a **sum over the 100 test engines** and is dominated by a few large late errors. Its seed-to-seed
  spread is large (e.g. local whole-engine 652 ± 80), so it should be reported with its SD and never on its own.

### 10.4 Communication and runtime

**Communication rounds** (K=5, E=5; final-round checkpoint, which is the natural one for a round budget):
R=5 14.83 ± 0.94, R=10 13.78 ± 0.26, R=20 12.88 ± 0.30, R=30 13.33 ± 0.22. Accuracy improves up to R=20 and
then **does not improve further**; R=30 is slightly worse than R=20 in mean while costing 50% more traffic
(259.3 vs 172.9 MB). The selected checkpoint shows the same pattern (13.24 vs 13.48). Because E is fixed, more
rounds also means more total training (R×E passes), so this sweep mixes communication frequency with training length.

**Local epochs** (K=5, R=20): E=1 is clearly insufficient (17.42 ± 1.51 final; NASA 689 ± 269). E=3 (13.11 ± 0.24),
E=5 (12.88 ± 0.30) and E=10 (13.67 ± 0.39) are close, with E=5 lowest on average and E=10 worse than E=5 in mean.
E=3 and E=5 differ by less than one SD. More local computation at a fixed communication budget helps only up to a point.

**Number of clients, total data fixed** (R=20, E=5):
* IID window: K=2 13.62 ± 0.40, K=5 12.88 ± 0.30, K=10 13.53 ± 0.32 (final). **Not monotone**; K=5 is lowest.
  Each K changes the per-client shard size and the number of local steps per round, so this is not a pure
  "more participants" effect.
* Whole-engine: K=2 13.65 ± 0.41, K=5 13.87 ± 0.19, K=10 14.34 ± 0.59 (selected). Error **rises** as the same 90
  engines are split over more owners (45 → 18 → 9 engines per client).
* Traffic grows linearly with K: 69.1, 172.9, 345.7 MB (serialized-payload estimate).

**Communication accounting**: one model transfer = 864,260 bytes (862,724 B of fp32 weights + 1,536 B of `.npy`
headers). Primary setting (K=5, R=20): 86.43 MB up + 86.43 MB down = 172.85 MB (= 164.84 MiB). The original report's
"164.55 MB" was the same quantity in MiB without the headers. All figures are payload estimates; no network was measured.

**Runtime** (section 8b, seed 42, clean benchmark): with the same 100 passes over the data, centralized training took
78 s and FedAvg-IID 72 s of sequential compute on one RTX 4060 Laptop GPU, i.e. **the same training cost**.
FedAvg's derived parallel estimate is 15.8 s (each client ≈0.7 s per round). The original report's "FL 131 s vs
centralized 49 s" compared a fixed 100-pass FL budget with an early-stopped centralized run, plus Ray/Flower overhead.
In this simulation FedAvg's wall time is the sum of all client
work on one GPU. It should not be read as the latency of a real deployment, which would depend on parallel
clients, network bandwidth and stragglers.

**Accuracy–communication trade-off**: the cheapest FedAvg setting tested that stays within ≈1 cycle of the best is
R=10 (13.78 RMSE, 86 MB), half the traffic of R=20 (12.88, 173 MB). Centralized training needs no model exchange,
but the raw training data would have to be moved instead: the 90-engine training table is ≈1.3 MB as float32
(analytical estimate, `comm.raw_training_data_float32_bytes_estimate`), about 130× smaller than FedAvg's model
traffic at R=20. **FedAvg is not a bandwidth saving here.** Its motivation is data ownership and governance,
not communication.

### 10.5 Validation-based selection vs final checkpoint

For FedAvg the final-round model was better on the test set than the validation-selected one (12.88 vs 13.24), while
for centralized training selection helped (14.46 vs 14.82). The validation set has only 10 engines, so checkpoint
selection is noisy. Both checkpoints are reported everywhere. **The method ordering is the same under both
policies**, so the conclusions do not depend on the selection rule.

### 10.6 Limitations

1. **One dataset subset (FD001)**: single operating condition, single fault mode, 100 training engines. Results may
   not transfer to FD002–FD004 (multiple conditions / fault modes), where client heterogeneity is more realistic.
2. **Small validation set**: 10 engines (2,101 windows), fixed across seeds. Selection noise is visible (10.5).
   Varying the validation split across seeds, or cross-validation, would give a fuller variance estimate.
3. **Five seeds**: enough to show consistent orderings, but too few for a significance test at α = 0.05 (the
   smallest attainable two-sided sign-flip p is 0.0625). Effects smaller than about one SD (e.g. α = 1.0 vs IID) are not
   resolved.
4. **Untuned hyperparameters**: no method was tuned. The centralized baseline overfits under the shared budget,
   which favours FedAvg in the comparison.
5. **Simulated federation**: clients run sequentially in one process; full participation, no stragglers, no client
   dropout, no network. Communication is a serialized-payload estimate, not measured traffic.
6. **Server-side validation set**: FedAvg checkpoint selection uses a held-out set of engines at the server, which
   a real deployment may not have.
7. **Synthetic partitions**: the "airlines" are random partitions of one fleet simulated under one condition.
   Real airline heterogeneity (operating profiles, maintenance practice, sensor calibration) is not represented.
8. **No privacy guarantee**: raw data stay on clients, but model updates can leak information. No differential
   privacy or secure aggregation is implemented, so no formal privacy claim can be made.
9. **Test labels capped at 125** (common FD001 practice); the uncapped truth is stored in every
   `test_predictions.csv` for re-scoring.
