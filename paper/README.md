# paper/ — updated manuscript sections (v2 results)

The manuscript source is not in this repository, so the revised text is provided here as drop-in
LaTeX (Springer LNCS compatible) for you to paste into your manuscript (Overleaf / Word).

| File | Use |
|---|---|
| `updated_sections.tex` | Revised **abstract**, **Experimental Protocol**, **Results** (3 subsections, 3 tables, 5 figures), **Discussion & Limitations**, **Conclusion** |
| `figures/*.pdf` (+ `.png`) | Figures referenced by the .tex (vector PDF for LaTeX; PNG for Word) |
| `check_numbers.py` | Confirms every `mean ± SD` in the .tex exists in `results/v2/tables/` (`python paper/check_numbers.py`: 103/103 match) |

Needs `\usepackage{graphicx}` in the preamble. Upload `figures/` next to the .tex in Overleaf.
LaTeX was not compiled locally (no TeX installation on the build machine). Its structure was checked:
balanced braces, matched environments, no undefined references, all figure files present.

## What to replace in your current manuscript

| # | Old manuscript content | Replace with |
|---|---|---|
| 1 | Single-seed numbers (e.g. FL RMSE 12.95, centralized 16.01, gap 3.06) | Mean ± SD over 5 matched seeds, Table `tab:main` |
| 2 | "FL outperforms centralized learning" | Results §"Federated versus centralized training": gap 1.22 cycles, explained by regularisation from model averaging (K=1 ablation), no claim that FL is intrinsically better |
| 3 | (missing) local-only baseline | Results §"Federation versus local-only training": lead finding, 13.87 vs 16.28 RMSE, NASA 349 vs 652 |
| 4 | "R=30 is best" | R=20 is best; R=30 gives no further improvement (Table `tab:sweeps`) |
| 5 | "More clients improve accuracy" | Not monotone (IID); error rises with K under whole-engine ownership |
| 6 | Heterogeneity / engine-level discussion | Table `tab:part` + confound statements (quantity skew at α=0.1; engine ownership ≠ label heterogeneity) |
| 7 | "164.55 MB" (MiB, parameter-count estimate) | 172.85 MB serialized-payload estimate, MB = 10^6 bytes, no protocol overhead |
| 8 | Any privacy claim | Limitation (8): no formal privacy guarantee |
| — | Old protocol description | Section "Experimental Protocol" (matched budget, shared selection rule, seeds) |
| — | Old placeholder values (`results/FINAL_REPORT.md` §14) | `results/v2/paper_tables/PAPER_VALUES.md` |

**Checkpoint policy:** all manuscript tables use the validation-selected checkpoint. Final-checkpoint values,
which lead to the same conclusions, are in `results/v2/paper_tables/*_final.csv`.
(`results/v2/CONCLUSION_CHECK.md` quotes the final-round values for the round and epoch sweeps.
Both are correct; use the selected ones in the paper for consistency.)
