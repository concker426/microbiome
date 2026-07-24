# ProCyon v2 — Experiment Log (2026-07-24 05:03)

## Status: IDLE

## Result Files
- **procyon_v2_A1_pooling.json** (2026-07-15 13:47:02): ACC=
- **procyon_v2_A2_transformer.json** (2026-07-15 14:02:19): ACC=
- **procyon_v2_B1.json** (2026-07-15 14:23:00): ACC=
- **procyon_v2_B2.json** (2026-07-18 08:11:46): ACC=
- **procyon_v2_phase3.json** (2026-07-18 11:19:03): ACC=
- **procyon_v2_summary.json** (2026-07-15 14:23:00): ACC=

## Latest Output (bxsqj648f.output)
```
  A: The given gut microbiome composition includes several genera that have been implicated in Inflammatory Bowel Disease (IB...
  B: The classifier predicted the patient to be healthy with 100% confidence, likely due to the presence of certain genera th...
  C: The classifier predicted the patient to be healthy based on the relative abundances of various genera in their gut micro...

============================================================
EVALUATION METRICS
============================================================

  Raw genus list:
    Hallucinated genera:  0.3/response
    Input genera mentioned: 5.1/response
    Specificity ratio:    0.534 (higher = more specific)
    Prediction consistent: 30/50 (60%)

  SHAP only:
    Hallucinated genera:  0.0/response
    Input genera mentioned: 9.9/response
    Specificity ratio:    0.714 (higher = more specific)
    Prediction consistent: 49/50 (98%)

  SHAP + Literature:
    Hallucinated genera:  0.0/response
    Input genera mentioned: 9.8/response
    Specificity ratio:    0.665 (higher = more specific)
    Prediction consistent: 40/50 (80%)

Saved: /hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/phase45_validation.json
Saved: /hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/phase45_human_review.txt

PHASE 4.5 DONE
```

*Auto-generated at 2026-07-24 05:03:02*
