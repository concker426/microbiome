# ProCyon v2: Reproducibility Guide

## Quick Start

```bash
# 1. Train backbone + baselines + ablation + transfer (Tables 1-4)
python experiments/run_week1_experiments.py

# 2. Structural baselines (FT-Transformer, DeepSets)
python experiments/run_structural_baselines.py

# 3. LOO attribution reliability (Spearman, deletion, permutation)
python experiments/run_shap_reliability.py

# 4. Significance tests (McNemar, DeLong/bootstrap)
python experiments/run_significance_tests.py

# 5. Decontaminated cross-dataset transfer
python experiments/run_decontaminated_transfer.py

# 6. Calibration + error analysis + case studies
python experiments/run_paper_finalization.py

# 7. Figures
python experiments/run_dataset_analysis.py
python experiments/run_inductive_bias_figure.py

# 8. Build Overleaf package
bash experiments/prepare_overleaf_package.sh
```

## Environment

Python 3.10+, PyTorch 2.x, CUDA 12.x (GPU optional for most scripts).

```bash
pip install torch numpy scipy scikit-learn xgboost matplotlib transformers
```

## Data

- `data/qiita_ibd/clean_2538/`: 659 train, 167 test (AGP+FTP)
- `data/qiita_ibd/merged_all/`: 3350 train, 838 test (5 Qiita sources)

## Random Seeds

{42, 123, 456, 789, 1024}. Main result: 92.57% +/- 0.48% (5-seed ensemble).

## Key Results

| Experiment | Key Finding |
|-----------|-------------|
| Baseline | ProCyon v2 92.57% vs XGBoost 92.22% (n.s.) vs MGM 50.9% (p<0.001) |
| Ablation | Embedding +2.75% over raw; MLP essential |
| Embedding Dim | Saturates at E=512 (93.41%, 760K params) |
| Transfer | merged->clean 88.62% (decontaminated) |
| LOO Attribution | Spearman rho=0.72, deletion test 5.6x vs random |
| LLM Pilot | LOO grounding -> 0 hallucination, 98% consistency (n=50) |
| Calibration | ECE=0.05, Brier=0.056 |
