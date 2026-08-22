# Microbiome Classification Experiments

**Protocol**: Leakage-robust — 3 repeats × StratifiedGroupKFold(5) = 15 validation folds,
grouped by exact genus-sequence hash (identical inputs never cross folds).

---

## 1. Unified Baseline Comparison

**Table 1: Leakage-robust comparison on the clean_2538 IBD task.**
Identical genus-sequence inputs are kept in the same fold.
Values are mean ± standard deviation over 15 validation folds.

| Method | Accuracy | Macro-F1 | AUROC | AUPRC |
|--------|----------|----------|-------|-------|
| HistGradientBoosting | **0.925 ± 0.025** | **0.925 ± 0.025** | **0.979 ± 0.010** | **0.981 ± 0.010** |
| Extra Trees | 0.918 ± 0.024 | 0.918 ± 0.024 | 0.963 ± 0.013 | 0.958 ± 0.027 |
| Random Forest | 0.916 ± 0.023 | 0.915 ± 0.023 | 0.971 ± 0.013 | 0.970 ± 0.019 |
| **SimpleEmb + MLP (ours)** | **0.914 ± 0.020** | **0.913 ± 0.019** | **0.957 ± 0.014** | **0.952 ± 0.032** |
| RBF-SVM | 0.879 ± 0.032 | 0.877 ± 0.032 | 0.941 ± 0.023 | 0.942 ± 0.031 |
| MLP | 0.872 ± 0.031 | 0.871 ± 0.031 | 0.945 ± 0.024 | 0.950 ± 0.030 |
| Logistic Regression | 0.865 ± 0.024 | 0.864 ± 0.024 | 0.936 ± 0.017 | 0.940 ± 0.022 |
| MGM + MLP | 0.853 ± 0.028 | 0.852 ± 0.028 | 0.929 ± 0.026 | 0.924 ± 0.048 |
| MGM + Logistic Regression | 0.846 ± 0.026 | 0.845 ± 0.026 | 0.908 ± 0.019 | 0.911 ± 0.037 |
| MGM + Random Forest | 0.815 ± 0.031 | 0.813 ± 0.031 | 0.902 ± 0.025 | 0.912 ± 0.030 |

**Key findings:**
- 树模型是最强 classical baseline，HGB (0.925) 比 SimpleEmb (0.914) 高 1.1pp（噪音范围内）
- SimpleEmb 超过所有 MGM 变体 6.1–9.9pp
- SimpleEmb 提供树模型没有的 embedding 空间（kNN/LOO/聚类/LLM 解释）

---

## 2. Embedding-Dimension Ablation

**Table 2: Embedding-dimension ablation under the exact input-grouped 3×5-fold CV protocol.**
Only the embedding dimension changes; all other architecture and optimization settings are fixed.

| E | Parameters | Accuracy | Macro-F1 | AUROC | AUPRC |
|---|------------|----------|----------|-------|-------|
| 16 | 24,994 | .8552 ± .0268 | .8543 ± .0265 | .9237 ± .0202 | .9180 ± .0409 |
| 32 | 48,706 | .8794 ± .0329 | .8782 ± .0335 | .9414 ± .0186 | .9442 ± .0224 |
| 64 | 96,130 | .9057 ± .0214 | .9050 ± .0214 | .9564 ± .0155 | .9543 ± .0309 |
| 128 | 190,978 | .9081 ± .0240 | .9072 ± .0241 | .9569 ± .0150 | .9557 ± .0264 |
| 256 | 380,674 | .9093 ± .0200 | .9086 ± .0199 | .9559 ± .0146 | .9525 ± .0374 |
| **512** | **760,066** | **.9141 ± .0197** | **.9134 ± .0195** | **.9572 ± .0142** | .9522 ± .0323 |
| 768 | 1,139,458 | **.9141 ± .0156** | **.9134 ± .0156** | .9571 ± .0149 | .9519 ± .0354 |

**结论**: E=512 饱和，768 无额外收益（参数 +50%）。

---

## 3. Representation and Learning-Curve Ablations

**Table 3: Controlled ablation of token representation and classifier head
under the exact input-grouped 3×5-fold CV protocol.**

| Model | Parameters | Accuracy | Macro-F1 | AUROC | AUPRC |
|-------|------------|----------|----------|-------|-------|
| Presence+Linear | 2,454 | .8756 ± .0370 | .8735 ± .0370 | .9494 ± .0194 | .9570 ± .0178 |
| Presence+MLP | 315,138 | **.9117 ± .0247** | **.9108 ± .0248** | **.9688 ± .0115** | **.9735 ± .0115** |
| Embedding64+Linear | 78,594 | .8566 ± .0364 | .8537 ± .0366 | .9163 ± .0285 | .9090 ± .0490 |
| Embedding64+MLP | 96,130 | .9057 ± .0214 | .9050 ± .0214 | .9564 ± .0155 | .9543 ± .0309 |

**关键发现**: Presence+MLP (0.912) ≈ Embedding64+MLP (0.906) — embedding 对准确率贡献有限，
其价值在可解释性空间。Linear head 两种条件下都失败 — 非线性必要。

**Table 4: Nested grouped-CV learning curves.**
The training fraction is sampled from only the outer training groups.

| Train fraction | HGB AUROC | Presence+MLP AUROC | Embedding64+MLP AUROC |
|----------------|-----------|--------------------|-----------------------|
| 20% | .9222 ± .0272 | .9404 ± .0185 | .8995 ± .0269 |
| 40% | .9619 ± .0117 | .9592 ± .0114 | .9290 ± .0216 |
| 60% | .9719 ± .0100 | .9647 ± .0106 | .9467 ± .0153 |
| 80% | .9763 ± .0089 | .9663 ± .0101 | .9508 ± .0167 |
| 100% | **.9791 ± .0097** | .9688 ± .0115 | .9564 ± .0155 |

---

## 4. Out-of-Fold Case Analysis

**Table 5: Representative out-of-fold cases under the exact input-grouped split (seed 42).**
Values are disease probabilities. Token IDs are shown without taxonomic names because
the encoded token-to-genus mapping has not yet been validated.

| Case | Sample ID | Truth | Active tokens (prefix) | SimpleEmb+MLP | HGB | MGM+MLP |
|------|-----------|-------|------------------------|---------------|-----|---------|
| SimpleEmb correct; HGB wrong | 2538.1000715 | Healthy | 86: 400, 37, 83, 200, 1184, 1063, ... | H (0.0028) | D (0.8832) | D (0.8867) |
| HGB correct; SimpleEmb wrong | 2538.1003420 | Disease | 86: 200, 1184, 1153, 531, 1061, 1020, ... | H (0.1042) | D (0.9973) | H (0.1749) |
| SimpleEmb correct; MGM wrong | 2538.1002401 | Disease | 4: 200, 818, 1061, 3 | D (1.0000) | D (0.7494) | H (0.3979) |
| High-confidence false positive | 2538.1000719 | Healthy | 1: 3 | D (1.0000) | D (0.8243) | D (1.0000) |
| High-confidence false negative | 2538.1003229 | Disease | 2: 1217, 3 | H (0.0000) | D (0.7492) | D (0.8885) |
| All correct | 2538.1003905 | Disease | 23: 1173, 531, 200, 589, 818, 1059, ... | D (1.0000) | D (0.9999) | D (1.0000) |

**关键发现:**
- SimpleEmb 独有优势：识别 HGB/MGM 都误判的 healthy 样本；处理 4-token 极端低丰度 IBD（MGM 失败）
- HGB 独有优势：恢复 SimpleEmb 漏掉的 borderline disease
- 共同失败：1-token 假阳性（疑似测序伪影）骗过所有模型；2-token 假阴性仅 SimpleEmb 漏掉
- 这些极端低丰度样本对应 Cluster 1 亚型，是系统性失败模式
