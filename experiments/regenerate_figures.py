#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate dataset/architecture figures for the SimpleEmb paper.
Produces two clean, large-font figures:
  1. dataset_architecture_figure.png  (SimpleEmb vs MGM architecture)
  2. dataset_statistics_figure.png    (6 dataset-statistics panels)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

# ── global font sizes: render comparable to 11pt body text ──
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'font.family': 'DejaVu Sans',
})

print("=" * 60)
print("REGENERATE DATASET + ARCHITECTURE FIGURES")
print("=" * 60)

# ── Load data ──
train_data = [json.loads(l) for l in open(f'{DATA_DIR}/train_nl.jsonl')]
test_data = [json.loads(l) for l in open(f'{DATA_DIR}/test_nl.jsonl')]
ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    info = json.load(f)
    GENUS_NAMES = info['genus_names']

all_seqs = np.concatenate([ts, xs])
all_masks = np.concatenate([tm, xm])
all_data = train_data + test_data
all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in all_data])

# ── per-sample richness + genus prevalence ──
genus_counts = all_masks.sum(axis=1)
genus_prevalence = {}
for i in range(len(all_seqs)):
    valid = all_masks[i].astype(bool)
    seen = set()
    for j in range(len(all_seqs[i])):
        if valid[j] and all_seqs[i, j] > 0:
            gid = int(all_seqs[i, j])
            if gid not in seen:
                gname = GENUS_NAMES[gid - 1] if gid - 1 < len(GENUS_NAMES) else f'g{gid}'
                genus_prevalence[gname] = genus_prevalence.get(gname, 0) + 1
                seen.add(gid)

prevalence_sorted = sorted(genus_prevalence.items(), key=lambda x: x[1], reverse=True)
n_total = len(all_data)

# ═══════════════════════════════════════════════════════════
# FIGURE 1: architecture comparison (SimpleEmb vs MGM)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 24); ax.set_ylim(0, 10); ax.axis('off')

def box(ax, x, y, w, h, text, color, fs=11, bold=False):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.25",
                       facecolor=color, edgecolor='black', linewidth=1.4, alpha=0.92)
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=('bold' if bold else 'normal'))

def arrow(ax, x1, y1, x2, y2, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color='#222222'))

# ── MGM (left) ──
ax.text(5.5, 9.55, 'MGM (pretrained foundation model)', fontsize=13, fontweight='bold',
        ha='center', color='#B71C1C')
box(ax, 5.5, 8.3, 5.2, 0.85, 'Genus abundance profile', '#FFCDD2', 11)
arrow(ax, 5.5, 7.85, 5.5, 7.35)
box(ax, 5.5, 6.85, 5.2, 0.85, 'Token embedding (768-d)', '#FFCDD2', 11)
arrow(ax, 5.5, 6.4, 5.5, 5.9)
box(ax, 5.5, 5.4, 5.2, 0.85, '8-layer Transformer (34M)', '#EF9A9A', 11)
arrow(ax, 5.5, 4.95, 5.5, 4.45)
box(ax, 5.5, 3.95, 5.2, 0.85, 'Attention pooling', '#EF9A9A', 11)
arrow(ax, 5.5, 3.5, 5.5, 3.0)
box(ax, 5.5, 2.5, 5.2, 0.85, 'MLP classifier', '#E57373', 11)
arrow(ax, 5.5, 2.05, 5.5, 1.55)
box(ax, 5.5, 1.05, 5.2, 0.85, '"IBD" / "Healthy"', '#B71C1C', 11, bold=True)
ax.text(5.5, 0.15, '34M params  |  pretrained on 263k samples  |  50.9% ACC (below majority)',
        fontsize=10.5, ha='center', color='#B71C1C', style='italic')

# ── SimpleEmb (right) ──
ax.text(18.5, 9.55, 'SimpleEmb (ours)', fontsize=13, fontweight='bold',
        ha='center', color='#1B5E20')
box(ax, 18.5, 8.3, 5.4, 0.85, 'Genus abundance profile', '#C8E6C9', 11)
arrow(ax, 18.5, 7.85, 18.5, 7.35)
box(ax, 18.5, 6.85, 5.4, 0.85, 'nn.Embedding(1226, 512)', '#A5D6A7', 11)
arrow(ax, 18.5, 6.4, 18.5, 5.9)
box(ax, 18.5, 5.4, 5.4, 0.85, 'Masked mean pooling  (h, 512-d)', '#A5D6A7', 11)

# branch
box(ax, 18.5, 4.35, 1.6, 0.6, 'h', '#81C784', 12, bold=True)
arrow(ax, 17.7, 4.2, 14.2, 3.4)
arrow(ax, 19.3, 4.2, 22.6, 3.4)
box(ax, 14.2, 2.85, 4.6, 0.95, 'MLP classifier\n(512-256-2)', '#66BB6A', 11)
box(ax, 22.6, 2.85, 4.6, 0.95, 'LOO attribution\n+ Qwen2-7B explanation', '#4CAF50', 10.5)
arrow(ax, 14.2, 2.3, 14.2, 1.7)
arrow(ax, 22.6, 2.3, 22.6, 1.7)
box(ax, 14.2, 1.1, 4.0, 0.85, '"IBD" / "Healthy"', '#1B5E20', 11, bold=True)
box(ax, 22.6, 1.1, 4.6, 0.85, 'Per-genus evidence', '#1B5E20', 11, bold=True)
ax.text(18.5, 0.15, '1.1M params  |  no pretraining  |  92.57% ACC  |  LOO + LLM interpretation',
        fontsize=10.5, ha='center', color='#1B5E20', style='italic')

fig.tight_layout(pad=0.4)
fig.savefig(f'{OUT_DIR}/dataset_architecture_figure.png', dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"Saved architecture figure -> {OUT_DIR}/dataset_architecture_figure.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: dataset statistics (2 x 3 grid)
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.5, 7.2))

# B: class distribution
ax = fig.add_subplot(2, 3, 1)
train_d = sum(1 for d in train_data if d['label'] == 'Disease')
train_h = sum(1 for d in train_data if d['label'] == 'Healthy')
test_d = sum(1 for d in test_data if d['label'] == 'Disease')
test_h = sum(1 for d in test_data if d['label'] == 'Healthy')
x = [0, 1]; w = 0.35
ax.bar([xi - w/2 for xi in x], [train_h, test_h], w, label='Healthy', color='#4CAF50')
ax.bar([xi + w/2 for xi in x], [train_d, test_d], w, label='Disease (IBD)', color='#F44336')
for i, (h, d) in enumerate(zip([train_h, test_h], [train_d, test_d])):
    ax.text(i - w/2, h + 4, str(h), ha='center', fontsize=11, fontweight='bold')
    ax.text(i + w/2, d + 4, str(d), ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Train', 'Test'])
ax.set_ylabel('Number of samples')
ax.legend(fontsize=9)
ax.set_title('A. Class distribution', fontweight='bold', loc='left')

# C: prevalence distribution
ax = fig.add_subplot(2, 3, 2)
prevalence_vals = [v / n_total for g, v in prevalence_sorted]
ax.hist(prevalence_vals, bins=50, color='#1565C0', alpha=0.8)
ax.axvline(x=0.5, color='#F44336', linestyle='--', linewidth=1.4, label='50% prevalence')
ax.axvline(x=0.05, color='#FF9800', linestyle='--', linewidth=1.4, label='5% (rare)')
ax.set_xlabel('Prevalence (fraction of samples)')
ax.set_ylabel('Number of genera')
ax.legend(fontsize=9)
ax.set_title('B. Genus prevalence', fontweight='bold', loc='left')

# D: richness
ax = fig.add_subplot(2, 3, 3)
ax.hist(genus_counts[all_labels == 0], bins=20, alpha=0.7, label='Healthy', color='#4CAF50')
ax.hist(genus_counts[all_labels == 1], bins=20, alpha=0.7, label='Disease (IBD)', color='#F44336')
ax.set_xlabel('Genera per sample')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)
ax.set_title('C. Richness distribution', fontweight='bold', loc='left')

# E: top-30 prevalent genera
ax = fig.add_subplot(2, 3, 4)
top30 = prevalence_sorted[:30]
names = [g[:16] for g, _ in top30]
vals = [v / n_total for _, v in top30]
colors = ['#1565C0' if v > n_total * 0.5 else '#90CAF9' for _, v in top30]
ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('Prevalence'); ax.invert_yaxis()
ax.set_title('D. Top-30 prevalent genera', fontweight='bold', loc='left')

# F: long-tail
ax = fig.add_subplot(2, 3, 5)
cumsum = np.cumsum([v for _, v in prevalence_sorted])
cumsum_norm = cumsum / cumsum[-1]
n_genera = len(prevalence_sorted)
ax.plot(range(n_genera), cumsum_norm, color='#1565C0', linewidth=2)
ax.axhline(y=0.5, color='#F44336', linestyle='--', alpha=0.7)
ax.axhline(y=0.9, color='#FF9800', linestyle='--', alpha=0.7)
n50 = np.searchsorted(cumsum_norm, 0.5) + 1
n90 = np.searchsorted(cumsum_norm, 0.9) + 1
ax.annotate(f'{n50} genera cover 50%', (n50, 0.5), fontsize=10,
            xytext=(n50 + 15, 0.38), arrowprops=dict(arrowstyle='->', color='#F44336'))
ax.annotate(f'{n90} genera cover 90%', (n90, 0.9), fontsize=10,
            xytext=(n90 + 15, 0.78), arrowprops=dict(arrowstyle='->', color='#FF9800'))
ax.set_xlabel('Genus rank (by prevalence)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('E. Long-tail distribution', fontweight='bold', loc='left')

# G: rarity categories
ax = fig.add_subplot(2, 3, 6)
rare_count = sum(1 for _, v in prevalence_sorted if v < n_total * 0.01)
mid_count = sum(1 for _, v in prevalence_sorted if n_total * 0.01 <= v < n_total * 0.1)
common_count = sum(1 for _, v in prevalence_sorted if n_total * 0.1 <= v < n_total * 0.5)
ubiq_count = sum(1 for _, v in prevalence_sorted if v >= n_total * 0.5)
sizes = [rare_count, mid_count, common_count, ubiq_count]
labels = [f'Rare (<1%)\n{rare_count}', f'Low (1-10%)\n{mid_count}',
          f'Common (10-50%)\n{common_count}', f'Ubiquitous (>50%)\n{ubiq_count}']
ax.pie(sizes, labels=labels, colors=['#FFCDD2', '#FFAB91', '#81D4FA', '#1565C0'],
       startangle=90, textprops={'fontsize': 10})
ax.set_title('F. Genus rarity categories', fontweight='bold', loc='left')

fig.tight_layout(pad=0.8)
fig.savefig(f'{OUT_DIR}/dataset_statistics_figure.png', dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"Saved statistics figure -> {OUT_DIR}/dataset_statistics_figure.png")

# ── statistics json (unchanged schema) ──
stats = {
    'n_train': len(train_data), 'n_test': len(test_data),
    'n_total': n_total,
    'n_disease': int(all_labels.sum()), 'n_healthy': int(n_total - all_labels.sum()),
    'n_unique_genera': len(genus_prevalence),
    'mean_genera_per_sample': float(genus_counts.mean()),
    'std_genera_per_sample': float(genus_counts.std()),
    'rare_genera_pct_5': sum(1 for _, v in prevalence_sorted if v < n_total * 0.05),
    'ubiquitous_genera_pct_50': sum(1 for _, v in prevalence_sorted if v > n_total * 0.5),
    'top10_prevalent': [(g, v, v / n_total) for g, v in prevalence_sorted[:10]],
    'richness_healthy': float(genus_counts[all_labels == 0].mean()),
    'richness_disease': float(genus_counts[all_labels == 1].mean()),
    'cumulative_50pct_genera': int(np.searchsorted(cumsum_norm, 0.5) + 1),
    'cumulative_90pct_genera': int(np.searchsorted(cumsum_norm, 0.9) + 1),
}
with open(f'{OUT_DIR}/dataset_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved dataset_statistics.json")
print("DONE")
