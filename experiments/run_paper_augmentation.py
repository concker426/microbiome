#!/usr/bin/env python3
"""
Paper Augmentation: Learning Curve Figure + Foundation Model Comparison
========================================================================
1. Learning curve visualization from Aug 4 group-CV data
2. Foundation model comparison table (MGM/BiomeGPT/Waypoint/ProCyon v2)
3. Generate all figures and LaTeX snippets
"""
import json, os, sys, csv
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
RESULTS = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PAPER AUGMENTATION: Learning Curve + FM Comparison")
print("=" * 60)

# ═══════════════════════════════════════════
# 1. LEARNING CURVE
# ═══════════════════════════════════════════
print("\n[1] Learning Curve from Group-CV data")

with open(f'{RESULTS}/decontaminated_groupcv_learning_curve_20260804/metrics_by_fold.csv') as f:
    rows = list(csv.DictReader(f))

# Aggregate by model x fraction
models = ['HistGradientBoosting', 'Presence_MLP', 'Embedding64_MLP']
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_curve = {}
for model in models:
    learning_curve[model] = {}
    for frac in fractions:
        subset = [r for r in rows if r['model'] == model and float(r['train_fraction']) == frac]
        accs = [float(r['accuracy']) for r in subset]
        aucs = [float(r['auroc']) for r in subset]
        learning_curve[model][frac] = {
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'n': len(subset),
        }
        print(f"  {model} frac={frac}: ACC={np.mean(accs):.4f}±{np.std(accs):.4f} AUC={np.mean(aucs):.4f}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: ACC vs fraction
ax = axes[0]
colors = {'HistGradientBoosting': '#FF9800', 'Presence_MLP': '#2196F3', 'Embedding64_MLP': '#1B5E20'}
labels = {'HistGradientBoosting': 'HistGB (classical)', 'Presence_MLP': 'MLP (no embedding)',
          'Embedding64_MLP': 'ProCyon v2 (E=64)'}
for model in models:
    fracs = list(learning_curve[model].keys())
    accs = [learning_curve[model][f]['acc_mean'] for f in fracs]
    stds = [learning_curve[model][f]['acc_std'] for f in fracs]
    ax.errorbar([f*100 for f in fracs], accs, yerr=stds, marker='o', markersize=6,
               label=labels[model], color=colors[model], linewidth=2, capsize=3)
ax.set_xlabel('Training Data (%)'); ax.set_ylabel('Group-CV Accuracy')
ax.set_title('A. Data Efficiency (Nested Group-CV)', fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel B: AUC vs fraction
ax = axes[1]
for model in models:
    fracs = list(learning_curve[model].keys())
    aucs = [learning_curve[model][f]['auc_mean'] for f in fracs]
    stds = [learning_curve[model][f]['auc_std'] for f in fracs]
    ax.errorbar([f*100 for f in fracs], aucs, yerr=stds, marker='s', markersize=6,
               label=labels[model], color=colors[model], linewidth=2, capsize=3)
ax.set_xlabel('Training Data (%)'); ax.set_ylabel('Group-CV AUROC')
ax.set_title('B. AUROC vs Training Data', fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/learning_curve.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/learning_curve.png")

# ═══════════════════════════════════════════
# 2. FOUNDATION MODEL COMPARISON TABLE
# ═══════════════════════════════════════════
print("\n[2] Foundation Model Comparison Table")

fm_comparison = {
    'MGM': {
        'venue': 'Advanced Science, 2025',
        'architecture': '6-layer Transformer (34M)',
        'granularity': 'Genus-level',
        'pretraining': 'Next-genus prediction, 263k samples',
        'interpretability': 'Attention + leave-one-genus-out',
        'public_weights': 'Yes',
        'our_replication': 'ACC=50.9% (test split), 85.3% (group-CV)',
    },
    'BiomeGPT': {
        'venue': 'bioRxiv, 2026',
        'architecture': 'Transformer + dual embedding (species + abundance bin)',
        'granularity': 'Species-level',
        'pretraining': 'Masked abundance prediction, 13.3k samples',
        'interpretability': 'CLS attention analysis',
        'public_weights': 'No',
        'our_replication': 'Not possible (weights unavailable)',
    },
    'Waypoint/Atlas': {
        'venue': 'bioRxiv, 2026',
        'architecture': 'Transformer variants (6M-170M)',
        'granularity': 'Genus-level',
        'pretraining': '539k+ samples',
        'interpretability': 'Not emphasized',
        'public_weights': 'No',
        'our_replication': 'Not possible (weights unavailable)',
    },
    'ProCyon v2 (ours)': {
        'venue': 'This work',
        'architecture': 'SimpleEmb + MLP (1.1M), no Transformer',
        'granularity': 'Genus-level',
        'pretraining': 'None (random init)',
        'interpretability': 'LOO attribution + LLM explanation',
        'public_weights': 'Yes',
        'our_replication': 'ACC=92.57% (test), 91.41% (group-CV)',
    },
}

# LaTeX table
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Comparison of microbiome foundation models.}")
tex.append(r"ProCyon v2 achieves competitive performance with 30$\times$ fewer parameters")
tex.append(r"and no pretraining, while providing LOO attribution and LLM-based interpretation.}")
tex.append(r"\label{tab:fm_comparison}")
tex.append(r"\begin{tabular}{p{2.5cm}p{2.5cm}p{2.2cm}p{2.2cm}p{3cm}p{1.5cm}}")
tex.append(r"\toprule")
tex.append(r"\textbf{Model} & \textbf{Architecture} & \textbf{Granularity} & \textbf{Pretraining} & \textbf{Interpretability} & \textbf{Weights} \\")
tex.append(r"\midrule")
for name, info in fm_comparison.items():
    tex.append(f"  {name} & {info['architecture']} & {info['granularity']} & {info['pretraining']} & {info['interpretability']} & {info['public_weights']} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")

tex.append(r"\vspace{4pt}")
tex.append(r"\begin{tabular}{ll}")
tex.append(r"\toprule")
tex.append(r"\textbf{Model} & \textbf{IBD/Healthy classification (our evaluation)} \\")
tex.append(r"\midrule")
tex.append(r"  MGM & 50.9\% (held-out test), 85.3\% (group-CV) \\")
tex.append(r"  BiomeGPT & N/A (weights not public; reported 0.921 AUROC 10-fold CV) \\")
tex.append(r"  Waypoint/Atlas & N/A (weights not public) \\")
tex.append(r"  \textbf{ProCyon v2} & \textbf{92.57\% (held-out test), 91.41\% (group-CV)} \\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

with open(f'{OUT_DIR}/fm_comparison_table.tex', 'w') as f:
    f.write('\n'.join(tex))
print(f"Saved: {OUT_DIR}/fm_comparison_table.tex")

# ═══════════════════════════════════════════
# 3. COMBINED FIGURE: Attribution Biology + Study Effect
# ═══════════════════════════════════════════
print("\n[3] Attribution Biology Figure")

with open(f'{OUT_DIR}/attribution_biology_study.json') as f:
    bio_results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: LOO vs fold-change scatter
ax = axes[0]
# Recompute scatter data
import pickle
with open(f'{OUT_DIR}/shap_data_full.pkl', 'rb') as f:
    shap_by_id = {}
    for s in pickle.load(f)['all_samples']:
        shap_by_id[s['sample_id']] = s['importance']

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES = json.load(f)['genus_names']

# LOO importance
global_loo = {}
for s in shap_by_id.values():
    for g in s:
        gn = g['genus_name']
        if gn not in global_loo:
            global_loo[gn] = []
        global_loo[gn].append(abs(g['importance']))

# fold change (recompute quickly)
all_seqs = np.load('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/train_genus_sequences.npy')
all_seqs = np.concatenate([all_seqs, np.load('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/test_genus_sequences.npy')])
all_masks = np.load('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/train_genus_masks.npy')
all_masks = np.concatenate([all_masks, np.load('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/test_genus_masks.npy')])
train_data = [json.loads(l) for l in open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/train_nl.jsonl')]
test_data = [json.loads(l) for l in open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538/test_nl.jsonl')]
all_labels = np.array([1 if d['label']=='Disease' else 0 for d in train_data+test_data])

V = 1226
abundance = np.zeros((len(all_labels), V), dtype=np.float32)
for i in range(len(all_labels)):
    valid = all_masks[i].astype(bool)
    for j in range(len(all_seqs[i])):
        if valid[j] and all_seqs[i][j] > 0:
            abundance[i, int(all_seqs[i][j])] += 1
    t = abundance[i].sum()
    if t > 0: abundance[i] /= t

eps = 1e-6
lfc = np.abs(np.log2((abundance[all_labels==1].mean(0) + eps) / (abundance[all_labels==0].mean(0) + eps)))

x_vals = []; y_vals = []
for gn, imps in global_loo.items():
    if gn in GENUS_NAMES:
        gid = GENUS_NAMES.index(gn) + 1
        if abundance[:, gid].astype(bool).sum() >= 50:
            x_vals.append(lfc[gid])
            y_vals.append(np.mean(imps))
from scipy.stats import spearmanr
rho, p = spearmanr(x_vals, y_vals)
ax.scatter(x_vals, y_vals, s=20, alpha=0.5, c='#1565C0', edgecolors='none')
ax.set_xlabel('|log2 fold change| (Disease vs Healthy)')
ax.set_ylabel('Mean |LOO Importance|')
ax.set_title(f'A. Attribution vs Differential Abundance\n(ρ={rho:.3f}, p={p:.3f}, n={len(x_vals)}, prev≥50)', fontweight='bold', loc='left', fontsize=9)

# Panel B: Disease structure in embedding
ax = axes[1]
bio_b = bio_results['exp_b_study_effect']
bars = [bio_b['intra_disease_cosine'], bio_b['intra_healthy_cosine']]
names = ['Disease-Disease', 'Healthy-Healthy']
ax.bar(names, bars, color=['#F44336', '#4CAF50'], edgecolor='none')
for i, v in enumerate(bars):
    ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Intra-class Cosine Similarity')
ax.set_title('B. Embedding Structure: Disease vs Healthy\n(IBD more heterogeneous)', fontweight='bold', loc='left', fontsize=9)
ax.set_ylim(0, 0.8); ax.grid(True, alpha=0.3, axis='y')

# Panel C: Learning curve summary
ax = axes[2]
for model in models:
    fracs = list(learning_curve[model].keys())
    accs = [learning_curve[model][f]['acc_mean'] for f in fracs]
    ax.plot([f*100 for f in fracs], accs, marker='o', markersize=5,
           label=labels[model], color=colors[model], linewidth=2)
ax.set_xlabel('Training Data (%)'); ax.set_ylabel('Group-CV Accuracy')
ax.set_title('C. Data Efficiency\n(SimpleEmb works with limited data)', fontweight='bold', loc='left', fontsize=9)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/attribution_biology_figure.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/attribution_biology_figure.png")

# Save learning curve results
with open(f'{OUT_DIR}/learning_curve.json', 'w') as f:
    json.dump(learning_curve, f, indent=2)
print(f"Saved: {OUT_DIR}/learning_curve.json")
print("\nDONE")
