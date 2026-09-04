# ProCyon v2 — Experiment Log (2026-09-04 10:03)

## Status: IDLE

## Result Files
- **procyon_v2_A1_pooling.json** (2026-07-15 13:47:02): ACC=
- **procyon_v2_A2_transformer.json** (2026-07-15 14:02:19): ACC=
- **procyon_v2_B1.json** (2026-07-15 14:23:00): ACC=
- **procyon_v2_B2.json** (2026-07-18 08:11:46): ACC=
- **procyon_v2_phase3.json** (2026-07-18 11:19:03): ACC=
- **procyon_v2_summary.json** (2026-07-15 14:23:00): ACC=

## Latest Output (bltr1al1l.output)
```
  [160/167] processed...

Completed 167 samples.

======================================================================
LLM EXPLANATION BENCHMARK RESULTS
======================================================================
Metric                              Raw LLM     SHAP+LLM SHAP+Lit+LLM
----------------------------------------------------------------------
Genera mentioned                     5.4671       9.0240       9.0479
Hallucinations                       0.7246       0.3593       0.3533
SHAP Consistency                     0.6992       0.9915       0.9952
Direction Correct                    0.2692       0.9363       0.9309
Pred Consistent                      0.6347       0.9820       0.9760
Specificity                          0.3788       0.4412       0.3316
Lit Consistent                       0.0909       1.0000       1.0000

============================================================
SELECTING CASE STUDIES
============================================================
  Case 1: 2538.1002713 (correct + good SHAP explanation)
  Case 2: 2538.1000031 (prob=0.555)
  Case 3: 2538.1003420 (true=Disease, pred=HEALTHY)
  Case 4: 2538.1002713 (prob=1.000)

Saved: /hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/llm_benchmark_results.json
Saved: /hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/case_studies.txt
Saved: /hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/llm_benchmark_table.tex

EXP 6 DONE
```

*Auto-generated at 2026-09-04 10:03:01*
