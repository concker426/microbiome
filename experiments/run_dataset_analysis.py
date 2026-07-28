#!/usr/bin/env python3
"""
Dataset Analysis + Architecture Figure for ProCyon v2 Paper
============================================================
1. Dataset statistics (class balance, genus prevalence, rarity, long-tail)
2. Architecture comparison diagram (ProCyon v2 vs MGM)
3. training data distribution analysis
"""
import json, os, sys
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("DATASET ANALYSIS + ARCHITECTURE FIGURE")
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
all_labels = np.array([1 if d['label']=='Disease' else 0 for d in all_data])

# ═══════════════════════════════════════════
# 1. DATASET STATISTICS
# ═══════════════════════════════════════════
print("\n[1] Dataset Statistics")

# Per-sample genus count
genus_counts = all_masks.sum(axis=1)
# Genus prevalence (in how many samples does each genus appear)
genus_prevalence = {}
for i in range(len(all_seqs)):
    valid = all_masks[i].astype(bool)
    seen = set()
    for j in range(len(all_seqs[i])):
        if valid[j] and all_seqs[i, j] > 0:
            gid = int(all_seqs[i, j])
            if gid not in seen:
                gname = GENUS_NAMES[gid-1] if gid-1 < len(GENUS_NAMES) else f'g{gid}'
                genus_prevalence[gname] = genus_prevalence.get(gname, 0) + 1
                seen.add(gid)

# Sort by prevalence
prevalence_sorted = sorted(genus_prevalence.items(), key=lambda x: x[1], reverse=True)
n_total = len(all_data)

print(f"  Total samples: {n_total} (Disease={all_labels.sum()}, Healthy={n_total-all_labels.sum()})")
print(f"  Unique genera across all samples: {len(genus_prevalence)}")
print(f"  Mean genera per sample: {genus_counts.mean():.1f} ± {genus_counts.std():.1f}")
print(f"  Top-5 most prevalent: {[(g, v, f'{v/n_total:.1%}') for g, v in prevalence_sorted[:5]]}")
print(f"  Rare genera (<5% prevalence): {sum(1 for g, v in prevalence_sorted if v < n_total*0.05)}")
print(f"  Genera in >50% samples: {sum(1 for g, v in prevalence_sorted if v > n_total*0.5)}")

# ═══════════════════════════════════════════
# 2. VISUALIZATION
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))

# ── Panel A: Architecture Comparison ──
ax = fig.add_subplot(3, 3, (1, 3))  # Span top row, 2 columns
ax.set_xlim(0, 20); ax.set_ylim(0, 14); ax.axis('off')
ax.set_title('Architecture Comparison: MGM vs ProCyon v2', fontsize=13, fontweight='bold', loc='left')

def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.3",
                          facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)

def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# MGM side (left)
ax.text(4, 13.5, 'MGM (Ning et al., 2024)', fontsize=11, fontweight='bold', ha='center', color='#B71C1C')
draw_box(ax, 4, 12.5, 4.5, 0.8, 'Genus Abundance Profile\n[g1, g2, ..., g86]', '#FFCDD2', 7)
draw_arrow(ax, 4, 12.1, 4, 11.5)
draw_box(ax, 4, 11, 4.5, 0.8, 'Token Embedding\n(1226, 768) × 86', '#FFCDD2', 7)
draw_arrow(ax, 4, 10.6, 4, 10)
draw_box(ax, 4, 9.5, 4.5, 0.8, '6× Transformer Layer\n(Self-Attention + FFN)', '#EF9A9A', 7)
draw_arrow(ax, 4, 9.1, 4, 8.5)
draw_box(ax, 4, 8, 4.5, 0.8, 'Attention Pooling\n(learned query)', '#EF9A9A', 7)
draw_arrow(ax, 4, 7.6, 4, 7)
draw_box(ax, 4, 6.5, 4.5, 0.8, 'MLP Classifier\n(768→256→2)', '#E57373', 7)
draw_arrow(ax, 4, 6.1, 4, 5.5)
draw_box(ax, 4, 5, 4, 0.8, '"IBD" / "Healthy"', '#B71C1C', 9, bold=True)

# MGM stats
ax.text(4, 4.2, '34M params | Pretrained 263k samples\nACC: 50.9% on clean_2538', fontsize=8,
        ha='center', color='#B71C1C', style='italic')

# ProCyon v2 side (right)
ax.text(16, 13.5, 'ProCyon v2 (this work)', fontsize=11, fontweight='bold', ha='center', color='#1B5E20')
draw_box(ax, 16, 12.5, 4.5, 0.8, 'Genus Abundance Profile\n[g1, g2, ..., g86]', '#C8E6C9', 7)
draw_arrow(ax, 16, 12.1, 16, 11.5)
draw_box(ax, 16, 11, 4.5, 0.8, 'SimpleEmb\nnn.Embedding(1226, 768)', '#A5D6A7', 7)
draw_arrow(ax, 16, 10.6, 16, 10)
draw_box(ax, 16, 9.5, 4.5, 0.8, 'Masked Mean Pooling\n→ h ∈ ℝ⁷⁶⁸', '#A5D6A7', 7)
draw_arrow(ax, 16, 9.1, 16, 8.5)

# Branch point
draw_box(ax, 16, 8, 1.5, 0.5, 'h', '#81C784', 8, bold=True)

# Left branch: Classification
draw_arrow(ax, 15.3, 7.9, 12.5, 7)
draw_box(ax, 12.5, 6.5, 3.5, 0.8, 'MLP Classifier\n(768→256→BN→ReLU→Drop→2)', '#66BB6A', 7)
draw_arrow(ax, 12.5, 6.1, 12.5, 5.5)
draw_box(ax, 12.5, 5, 3, 0.8, '"IBD" / "Healthy"', '#1B5E20', 9, bold=True)
ax.text(12.5, 4.2, '92.57% ACC\n0.21M params', fontsize=8, ha='center', color='#1B5E20')

# Right branch: Explanation
draw_arrow(ax, 16.7, 7.9, 19.5, 7)
draw_box(ax, 19.5, 6.5, 3.5, 0.8, 'SHAP LOO Importance\nper-genus contribution', '#81C784', 7)
draw_arrow(ax, 19.5, 6.1, 19.5, 5.5)
draw_box(ax, 19.5, 5, 3.5, 0.8, 'Qwen2-7B-Instruct\nBiological Explanation', '#4CAF50', 7)
ax.text(19.5, 4.2, '0 hallucination\n98% consistency', fontsize=8, ha='center', color='#1B5E20')

# Cross-out annotations
ax.text(6.5, 9.5, '✗', fontsize=28, color='#B71C1C', ha='center', va='center', fontweight='bold')
ax.text(6.5, 8, '✗', fontsize=28, color='#B71C1C', ha='center', va='center', fontweight='bold')

# ── Panel B: Class Distribution ──
ax = fig.add_subplot(3, 3, 4)
train_d = sum(1 for d in train_data if d['label']=='Disease')
train_h = sum(1 for d in train_data if d['label']=='Healthy')
test_d = sum(1 for d in test_data if d['label']=='Disease')
test_h = sum(1 for d in test_data if d['label']=='Healthy')
x = [0, 1]; w = 0.35
ax.bar([xi-w/2 for xi in x], [train_h, test_h], w, label='Healthy', color='#4CAF50', edgecolor='none')
ax.bar([xi+w/2 for xi in x], [train_d, test_d], w, label='Disease (IBD)', color='#F44336', edgecolor='none')
for i, (h, d) in enumerate(zip([train_h, test_h], [train_d, test_d])):
    ax.text(i-w/2, h+5, str(h), ha='center', fontsize=10, fontweight='bold')
    ax.text(i+w/2, d+5, str(d), ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Train', 'Test'])
ax.set_ylabel('Number of Samples'); ax.legend()
ax.set_title(f'B. Class Distribution (clean_2538)\nTotal: {n_total} samples', fontweight='bold', loc='left')

# ── Panel C: Genus Prevalence Distribution ──
ax = fig.add_subplot(3, 3, 5)
prevalence_vals = [v/n_total for g, v in prevalence_sorted]
ax.hist(prevalence_vals, bins=50, color='#1565C0', edgecolor='none', alpha=0.8)
ax.axvline(x=0.5, color='#F44336', linestyle='--', linewidth=1.5, label='50% prevalence')
ax.axvline(x=0.05, color='#FF9800', linestyle='--', linewidth=1.5, label='5% (rare)')
ax.set_xlabel('Prevalence (fraction of samples)'); ax.set_ylabel('Number of Genera')
ax.set_title(f'C. Genus Prevalence Distribution\n{len(genus_prevalence)} unique genera', fontweight='bold', loc='left')
ax.legend(fontsize=7)

# ── Panel D: Genus Count per Sample ──
ax = fig.add_subplot(3, 3, 6)
ax.hist(genus_counts[all_labels==0], bins=20, alpha=0.7, label='Healthy', color='#4CAF50', edgecolor='none')
ax.hist(genus_counts[all_labels==1], bins=20, alpha=0.7, label='Disease (IBD)', color='#F44336', edgecolor='none')
ax.set_xlabel('Genera per Sample'); ax.set_ylabel('Frequency')
ax.set_title(f'D. Richness Distribution\nHealthy μ={genus_counts[all_labels==0].mean():.1f}, IBD μ={genus_counts[all_labels==1].mean():.1f}',
            fontweight='bold', loc='left')
ax.legend(fontsize=8)

# ── Panel E: Top-30 Genus Prevalence (bar chart) ──
ax = fig.add_subplot(3, 3, 7)
top30 = prevalence_sorted[:30]
names = [g[:15] for g, _ in top30]
vals = [v/n_total for _, v in top30]
colors = ['#1565C0' if v > n_total*0.5 else '#90CAF9' for _, v in top30]
ax.barh(range(len(names)), vals, color=colors, edgecolor='none')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
ax.set_xlabel('Prevalence'); ax.invert_yaxis()
ax.set_title('E. Top-30 Most Prevalent Genera', fontweight='bold', loc='left')

# ── Panel F: Long-tail (cumulative prevalence) ──
ax = fig.add_subplot(3, 3, 8)
cumsum = np.cumsum([v for _, v in prevalence_sorted])
cumsum_norm = cumsum / cumsum[-1]
n_genera = len(prevalence_sorted)
ax.plot(range(n_genera), cumsum_norm, color='#1565C0', linewidth=2)
ax.axhline(y=0.5, color='#F44336', linestyle='--', alpha=0.7)
ax.axhline(y=0.9, color='#FF9800', linestyle='--', alpha=0.7)
# Annotate
n50 = np.searchsorted(cumsum_norm, 0.5) + 1
n90 = np.searchsorted(cumsum_norm, 0.9) + 1
ax.annotate(f'{n50} genera\n→ 50% occurrences', (n50, 0.5), fontsize=8,
           xytext=(n50+20, 0.4), arrowprops=dict(arrowstyle='->', color='#F44336'))
ax.annotate(f'{n90} genera\n→ 90% occurrences', (n90, 0.9), fontsize=8,
           xytext=(n90+20, 0.8), arrowprops=dict(arrowstyle='->', color='#FF9800'))
ax.set_xlabel('Genus Rank (by prevalence)'); ax.set_ylabel('Cumulative Fraction')
ax.set_title(f'F. Long-tail Distribution\n{n50} genera cover 50%, {n90} cover 90%', fontweight='bold', loc='left')

# ── Panel G: Rare vs Common genera ──
ax = fig.add_subplot(3, 3, 9)
rare_count = sum(1 for _, v in prevalence_sorted if v < n_total*0.01)
mid_count = sum(1 for _, v in prevalence_sorted if n_total*0.01 <= v < n_total*0.1)
common_count = sum(1 for _, v in prevalence_sorted if n_total*0.1 <= v < n_total*0.5)
ubiq_count = sum(1 for _, v in prevalence_sorted if v >= n_total*0.5)
sizes = [rare_count, mid_count, common_count, ubiq_count]
labels = [f'Rare (<1%)\n{rare_count}', f'Low (1-10%)\n{mid_count}',
          f'Common (10-50%)\n{common_count}', f'Ubiquitous (>50%)\n{ubiq_count}']
colors_pie = ['#FFCDD2', '#FFAB91', '#81D4FA', '#1565C0']
wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie, startangle=90)
ax.set_title(f'G. Genus Rarity Categories\n({len(genus_prevalence)} total genera)', fontweight='bold', loc='left')

fig.suptitle('ProCyon v2: Dataset Analysis & Architecture', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUT_DIR}/dataset_architecture_figure.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/dataset_architecture_figure.png")

# ═══════════════════════════════════════════
# SAVE STATISTICS
# ═══════════════════════════════════════════
stats = {
    'n_train': len(train_data), 'n_test': len(test_data),
    'n_total': n_total,
    'n_disease': int(all_labels.sum()), 'n_healthy': int(n_total - all_labels.sum()),
    'n_unique_genera': len(genus_prevalence),
    'mean_genera_per_sample': float(genus_counts.mean()),
    'std_genera_per_sample': float(genus_counts.std()),
    'rare_genera_pct_5': sum(1 for _, v in prevalence_sorted if v < n_total*0.05),
    'ubiquitous_genera_pct_50': sum(1 for _, v in prevalence_sorted if v > n_total*0.5),
    'top10_prevalent': [(g, v, v/n_total) for g, v in prevalence_sorted[:10]],
    'richness_healthy': float(genus_counts[all_labels==0].mean()),
    'richness_disease': float(genus_counts[all_labels==1].mean()),
    'cumulative_50pct_genera': int(np.searchsorted(cumsum_norm, 0.5) + 1),
    'cumulative_90pct_genera': int(np.searchsorted(cumsum_norm, 0.9) + 1),
}
with open(f'{OUT_DIR}/dataset_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Saved: {OUT_DIR}/dataset_statistics.json")
print("DONE")
