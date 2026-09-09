# Experiment Log
Generated: 2026-09-09T11:59:33
Git commit: 68589b62171d
Software: Python=3.11.9 PyTorch=2.7.1+cu118 Flower=1.32.0 NumPy=1.26.4 sklearn=1.3.0
Hardware: CPU=Intel64 Family 6 Model 154 Stepping 3, GenuineIntel GPU=NVIDIA GeForce RTX 4060 Laptop GPU

---

## Exp A -- Dataset Statistics
**Time:** 2026-09-09T11:59:40
**Git:** 68589b62171d
**Config:** {'dataset': 'FD001'}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\dataset\\dataset_statistics.csv', 'D:\\federated_learning pipeline_rul\\results\\dataset\\dataset_statistics.txt']

---

## Exp B -- Model Statistics
**Time:** 2026-09-09T12:00:02
**Git:** 68589b62171d
**Config:** {}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\model\\model_statistics.csv']

---

## Exp C -- Centralized (cached)
**Time:** 2026-09-09T12:00:03
**Git:** 68589b62171d
**Config:** {'seeds': [42, 123, 2026]}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\centralized']

---

## Exp D -- FL-IID
**Time:** 2026-09-09T12:05:33
**Git:** 68589b62171d
**Config:** {'seeds': [42, 123, 2026]}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\fl_iid']

---

## Exp E -- Comparison
**Time:** 2026-09-09T12:05:33
**Git:** 68589b62171d
**Config:** {}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\comparisons\\centralized_vs_fl.csv']

---

## Exp F -- Rounds Sweep
**Time:** 2026-09-09T12:13:37
**Git:** 68589b62171d
**Config:** {'rounds': [5, 10, 20, 30, 40]}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\rounds']

---

## Exp G -- Epoch Sweep
**Time:** 2026-09-09T12:19:20
**Git:** 68589b62171d
**Config:** {'epochs': [1, 3, 5, 10]}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\local_epochs']

---

## Exp H -- Clients Sweep
**Time:** 2026-09-09T12:25:39
**Git:** 68589b62171d
**Config:** {'clients': [2, 5, 10]}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\clients']

---

## Exp I -- Non-IID Sweep
**Time:** 2026-09-09T12:35:46
**Git:** 68589b62171d
**Config:** {}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\non_iid']

---

## Exp J -- Engine-Level
**Time:** 2026-09-09T12:41:56
**Git:** 68589b62171d
**Config:** {}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\engine_level']

---

## Communication Estimates
**Time:** 2026-09-09T12:41:57
**Git:** 68589b62171d
**Config:** {'formula': 'C_FL=2*R*b*P*K', 'b': 4}
**Result files:** ['D:\\federated_learning pipeline_rul\\results\\communication\\communication_summary.csv']

---

