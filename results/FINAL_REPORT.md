# Federated Aircraft Engine Prognostics -- Results Report

Generated: 2026-09-09 12:41:57  |  Dataset: FD001  |  Primary seed: 42

---

## 1. Dataset

| Parameter | Value |
|---|---|
| dataset_name | NASA C-MAPSS |
| subset | FD001 |
| n_train_engines | 90 |
| n_val_engines | 10 |
| n_test_engines | 100 |
| n_train_windows | 15630 |
| n_val_windows | 2101 |
| n_test_samples | 100 |
| n_features | 17 |
| window_length | 30 |
| stride | 1 |
| rul_cap | 125 |
| removed_sensors | [1, 5, 6, 10, 16, 18, 19] |
| train_val_ratio | 90:10 |
| scaler_type | MinMaxScaler(0,1) fitted on train only |

---

## 2. Model Architecture & Statistics

| Parameter | Value |
|---|---|
| total_parameters | 215681 |
| trainable_parameters | 215681 |
| non_trainable_parameters | 0 |
| model_size_fp32_mb | 0.8228 |
| model_size_fp16_mb | 0.4114 |
| input_shape | (batch, 30, 17) |
| output_shape | (batch,) |
| inference_time_ms_per_sample | 0.012468 |

---

## 3. Centralized Baseline (seed=42)

| Metric | Value |
|---|---|
| RMSE | 16.0064 cycles |
| MAE  | 12.1683 cycles |
| NASA Score | 556.41 |
| Training time | 49.1 s |
| Best val loss | 160.540103 |
| Epochs | 24 |
| Inference time | 0.0533 ms/sample |

---

## 4. FL-IID Primary Results (seed=42)

| Metric | Value |
|---|---|
| Test RMSE | 12.9461 cycles |
| Test MAE  | 9.3372 cycles |
| Test NASA Score | 287.00 |
| Total training time | 131.3 s |
| Avg round time | -5.5 s |
| Clients | 5 |
| Rounds | 20 |
| Local epochs | 5 |

---

## 5. Communication Rounds Trade-off

| Rounds | RMSE | MAE | NASA | Comm (MB) | Avg Round (s) |
|---|---|---|---|---|---|
| 5 | 14.1436 | 10.7532 | 304.79 | 41.14 | -6.2 |
| 10 | 13.3187 | 9.9133 | 286.05 | 82.28 | -4.9 |
| 20 | 13.3589 | 9.7859 | 320.83 | 164.55 | -4.3 |
| 30 | 12.5027 | 9.0841 | 251.53 | 246.83 | -3.9 |
| 40 | 13.1960 | 9.7649 | 283.22 | 329.10 | -3.6 |

---

## 6. Local Epoch Sweep

| Local Epochs | RMSE | MAE | NASA | Time (s) | Comm (MB) |
|---|---|---|---|---|---|
| 1 | 19.5454 | 14.2195 | 1160.85 | 46.0 | 164.55 |
| 3 | 13.3809 | 9.5305 | 304.80 | 63.5 | 164.55 |
| 5 | 12.6514 | 9.2497 | 251.90 | 87.5 | 164.55 |
| 10 | 13.2997 | 9.7645 | 307.50 | 143.1 | 164.55 |

---

## 7. Client Scalability

| Clients | RMSE | MAE | NASA | Time (s) | Comm (MB) | Windows/Client |
|---|---|---|---|---|---|---|
| 2 | 15.1168 | 10.9148 | 456.97 | 100.5 | 65.82 | 7815 |
| 5 | 13.6066 | 10.1969 | 335.00 | 110.5 | 164.55 | 3126 |
| 10 | 13.0946 | 9.6970 | 274.50 | 164.5 | 329.10 | 1563 |

---

## 8. Non-IID / Data Heterogeneity

| Partition | RMSE | MAE | NASA | Time (s) |
|---|---|---|---|---|
| IID | 13.4501 | 9.9142 | 322.09 | 169.4 |
| 1.0 | 13.8361 | 10.2506 | 364.75 | 88.0 |
| 0.5 | 13.3822 | 9.8288 | 310.69 | 88.4 |
| 0.1 | 13.5834 | 10.6550 | 258.65 | 86.5 |

---

## 9. Engine-Level Client Ownership (seed=42)

| Metric | Value |
|---|---|
| Test RMSE | 14.6992 |
| Test MAE  | 11.0016 |
| Test NASA Score | 368.92 |
| Training time | 111.3 s |

---

## 10. Communication Cost Estimates

| Experiment | Clients | Rounds | Params | Download MB | Upload MB | Total MB |
|---|---|---|---|---|---|---|
| FL-IID (primary) | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Rounds-R5 | 5 | 5 | 215681 | 20.5689 | 20.5689 | 41.1379 |
| Rounds-R10 | 5 | 10 | 215681 | 41.1379 | 41.1379 | 82.2758 |
| Rounds-R20 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Rounds-R30 | 5 | 30 | 215681 | 123.4137 | 123.4137 | 246.8273 |
| Rounds-R40 | 5 | 40 | 215681 | 164.5515 | 164.5515 | 329.1031 |
| Epochs-E1 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Epochs-E3 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Epochs-E5 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Epochs-E10 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Clients-K2 | 2 | 20 | 215681 | 32.9103 | 32.9103 | 65.8206 |
| Clients-K5 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| Clients-K10 | 10 | 20 | 215681 | 164.5515 | 164.5515 | 329.1031 |
| NonIID-IID | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| NonIID-alpha1.0 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| NonIID-alpha0.5 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| NonIID-alpha0.1 | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |
| EngineLevel | 5 | 20 | 215681 | 82.2758 | 82.2758 | 164.5515 |

> NOTE: All communication values are ESTIMATES of parameter-transfer volume computed as C_FL = 2 x R x b x P x K (b=4 bytes FP32). They are NOT measured network traffic.

---

## 11. Computation / Training Time

| Experiment | Total Time (s) | Avg Round (s) | Inference (ms/sample) |
|---|---|---|---|
| Centralized_seed42 | 49.1 | NOT_RUN | 0.0533 |
| Centralized_seed123 | 30.2 | NOT_RUN | 0.0647 |
| Centralized_seed2026 | 53.0 | NOT_RUN | 0.0881 |
| FL_IID_seed42 | 131.3 | -5.5 | NOT_RUN |
| FL_IID_seed123 | 96.4 | -4.2 | NOT_RUN |
| FL_IID_seed2026 | 97.8 | -4.4 | NOT_RUN |
| Rounds_R5 | 38.8 | -6.2 | NOT_RUN |
| Rounds_R10 | 59.8 | -4.9 | NOT_RUN |
| Rounds_R20 | 96.1 | -4.3 | NOT_RUN |
| Rounds_R30 | 128.6 | -3.9 | NOT_RUN |
| Rounds_R40 | 157.4 | -3.6 | NOT_RUN |
| Epochs_E1 | 46.0 | -1.9 | NOT_RUN |
| Epochs_E3 | 63.5 | -2.8 | NOT_RUN |
| Epochs_E5 | 87.5 | -4.0 | NOT_RUN |
| Epochs_E10 | 143.1 | -6.8 | NOT_RUN |

---

## 12. Best Configurations

| Criterion | Configuration | RMSE | Comm MB |
|---|---|---|---|
| Best RMSE | Rounds-R30 | 12.5027 | 246.83 |
| Best MAE | Rounds-R30 | 12.5027 | 246.83 |
| Best NASA Score | Rounds-R30 | 12.5027 | 246.83 |
| Lowest Communication | Centralized | 16.0064 | 0.00 |
| Pareto Efficient | Centralized | 13.7475 | 0.00 |
| Pareto Efficient | Rounds-R10 | 13.3187 | 82.28 |
| Pareto Efficient | Rounds-R30 | 12.5027 | 246.83 |
| Pareto Efficient | Epochs-E5 | 12.6514 | 164.55 |

---

## 13. Key Findings

Based solely on actual experimental results:

1. FL-IID RMSE (12.9461) is 3.0603 cycles lower than centralized RMSE (16.0064).
2. Rounds sweep: R=5 RMSE=14.1436, R=40 RMSE=13.1960.
3. Local epoch sweep: E=1 RMSE=19.5454, E=10 RMSE=13.2997.
4. Client scalability: K=2 RMSE=15.1168, K=10 RMSE=13.0946.
5. Non-IID: alpha=0.1 RMSE=13.5834, IID RMSE=13.4501.
6. Engine-level RMSE=14.6992 vs FL-IID RMSE=12.9461.
7. Primary FL-IID communication estimate: 164.55 MB total (20 rounds, 5 clients).

> Statistical significance is NOT claimed without formal testing.
> Values above are descriptive from 3 seeds where multi-seed was run.

---

## 14. Paper Value Mapping (Section 26)

Use these values to fill placeholders in the Springer/LNCS manuscript.

### Section 3 -- Dataset & Methodology
```
[DATASET_NAME]            = NASA C-MAPSS
[SUBSET]                  = FD001
[N_TRAIN_ENGINES]         = 90
[N_VAL_ENGINES]           = 10
[N_TEST_ENGINES]          = 100
[N_TRAIN_WINDOWS]         = 15630
[N_VAL_WINDOWS]           = 2101
[N_TEST_SAMPLES]          = 100
[N_FEATURES]              = 17
[WINDOW_LENGTH]           = 30
[RUL_CAP]                 = 125
[REMOVED_SENSORS]         = [1, 5, 6, 10, 16, 18, 19]
```

### Section 3.3 -- Model
```
[PARAMETER_COUNT]         = 215681
[MODEL_SIZE_FP32_MB]      = 0.8228
[MODEL_SIZE_FP16_MB]      = 0.4114
[INPUT_SHAPE]             = (batch, 30, 17)
[INFERENCE_TIME_MS]       = 0.012468
```

### Section 5.1 -- Centralized vs FL
```
[CENT_RMSE]               = 16.0064
[CENT_MAE]                = 12.1683
[CENT_NASA]               = 556.41
[CENT_TIME_S]             = 49.1
[FL_RMSE]                 = 12.9461
[FL_MAE]                  = 9.3372
[FL_NASA]                 = 287.00
[FL_TIME_S]               = 131.3
```

### Section 5.2 -- Communication Rounds
```
[R5_RMSE]               = 14.1436
[R10_RMSE]               = 13.3187
[R20_RMSE]               = 13.3589
[R30_RMSE]               = 12.5027
[R40_RMSE]               = 13.1960
[R5_COMM_MB]            = 41.14
[R20_COMM_MB]            = 164.55
[R40_COMM_MB]            = 329.10
```

### Section 5.3 -- Local Epochs
```
[E1_RMSE]               = 19.5454
[E3_RMSE]               = 13.3809
[E5_RMSE]               = 12.6514
[E10_RMSE]               = 13.2997
[E1_TIME_S]               = 46.0
[E10_TIME_S]              = 143.1
```

### Section 5.4 -- Client Scalability
```
[K2_RMSE]               = 15.1168
[K5_RMSE]               = 13.6066
[K10_RMSE]               = 13.0946
[K2_COMM_MB]              = 65.82
[K10_COMM_MB]             = 329.10
```

### Section 5.5 -- Non-IID Heterogeneity
```
[IID_RMSE]                = 13.4501
[ALPHA_10_RMSE]           = 13.8361
[ALPHA_05_RMSE]           = 13.3822
[ALPHA_01_RMSE]           = 13.5834
```

### Section 5.6 -- Engine-Level
```
[ENGINE_RMSE]             = 14.6992
[ENGINE_MAE]              = 11.0016
[ENGINE_NASA]             = 368.92
```

### Section 5.7 -- Communication
```
[TOTAL_COMM_PRIMARY_MB]   = 164.5515
[MODEL_PARAMS]            = 215681
[COMM_FORMULA]            = C_FL = 2 x R x b x P x K  (b=4 bytes FP32)
```

### Section 5.8 -- Best Configuration
```
[BEST_RMSE_CONFIG]        = Rounds-R30
[BEST_COMM_CONFIG]        = Centralized
[PARETO_CONFIGS]          = Centralized; Rounds-R10; Rounds-R30; Epochs-E5
```

---

*End of FINAL_REPORT.md*