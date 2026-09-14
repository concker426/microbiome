#!/usr/bin/env python3
"""Generate inductive bias illustration figure for Section 4.5 — FIXED"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
exec(open("/hd/liujx/microbiome_llm_project/experiments/fig_style.py").read())
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

# ══════════════════════════════════════════════════════════
# Panel A: Inductive Bias Comparison
# ══════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(0, 20); ax.set_ylim(0, 16.5); ax.axis('off')
ax.set_title('A. Inductive Bias Comparison', fontsize=13, fontweight='bold', loc='left')

# Sequential (Transformer) side
ax.text(5, 15.8, 'Sequential (Transformer)', fontsize=12, fontweight='bold', ha='center', color='#B71C1C')
for i in range(6):
    x = 2 + i * 1.8; y = 14.5
    rect = FancyBboxPatch((x - 0.8, y - 0.5), 1.6, 1, boxstyle="round,pad=0.2",
                          facecolor='#FFCDD2', edgecolor='#B71C1C', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, f'g{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
for i in range(5):
    ax.annotate('', xy=(3.0 + i * 1.8, 14.5), xytext=(2.8 + i * 1.8, 14.5),
               arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2))
ax.text(8, 13.6, 'Position matters: g1→g2→g3...', ha='center', fontsize=9, color='#B71C1C', style='italic')
ax.text(8, 13.1, 'Self-attention over ordered sequence', ha='center', fontsize=9, color='#B71C1C', style='italic')

# Arrow down + results
ax.annotate('', xy=(8, 12.0), xytext=(8, 12.7),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(8, 11.7, 'FT-Transformer: 91.0% ACC', ha='center', fontsize=10, color='#B71C1C')
ax.text(8, 11.2, 'MGM (pretrained): 50.9% ACC', ha='center', fontsize=10, color='#B71C1C')

# Permutation-invariant (Set) side
ax.text(15, 15.8, 'Permutation-Invariant (Set)', fontsize=12, fontweight='bold', ha='center', color='#1B5E20')
centers = [(14.5, 13.7), (17, 13.0), (15.5, 12.3), (14, 11.7), (17.5, 14.3), (18.5, 12.5)]
for i, (cx, cy) in enumerate(centers):
    circle = plt.Circle((cx, cy), 0.6, facecolor='#C8E6C9', edgecolor='#1B5E20', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(cx, cy, f'g{i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

ax.text(15, 11.0, 'Position irrelevant: {g1, g2, ..., gk}', ha='center', fontsize=9, color='#1B5E20', style='italic')
ax.text(15, 10.5, 'Aggregate over SET, not sequence', ha='center', fontsize=9, color='#1B5E20', style='italic')

# Arrow down + results
ax.annotate('', xy=(15, 9.6), xytext=(15, 10.3),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(15, 9.3, 'DeepSets: 91.6% ACC', ha='center', fontsize=10, color='#1B5E20')
ax.text(15, 8.8, 'ProCyon v2: 91.6% ACC', ha='center', fontsize=10, color='#1B5E20', fontweight='bold')

# Plus annotation
ax.text(15, 8.1, '+ explicit embedding → SHAP, kNN, clustering, LLM', ha='center',
       fontsize=9, color='#1565C0', style='italic')

# Key insight box — with line spacing
rect = FancyBboxPatch((2, 3.5), 16, 3.8, boxstyle="round,pad=0.5",
                      facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5, alpha=0.5)
ax.add_patch(rect)
insight_lines = [
    'Key Insight: Microbiome abundance profiles are SETS, not sequences.',
    'Permutation-invariant models (DeepSets, SimpleEmb) match this structure.',
    'Transformers can work (FT: 91.0%) but carry unnecessary sequential bias.',
]
for k, line in enumerate(insight_lines):
    ax.text(10, 6.5 - k * 0.65, line, ha='center', fontsize=10, fontweight='bold', color='#1565C0')

# ══════════════════════════════════════════════════════════
# Panel B: Structural Baseline Results — FIXED
# ══════════════════════════════════════════════════════════
ax = axes[1]
methods = ['MGM\n(pretrained)', 'FT-Transformer\n(no pretrain)', 'DeepSets\n(set)', 'ProCyon v2\n(set+embed)']
accs = [50.9, 91.0, 91.6, 91.6]
aucs = [46.3, 95.5, 97.3, 96.4]
colors = ['#F44336', '#FF9800', '#4CAF50', '#1B5E20']

x = np.arange(len(methods)); w = 0.3
bars1 = ax.bar(x - w/2, accs, w, label='ACC (%)', color=colors, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + w/2, aucs, w, label='AUC (%)',
              color=colors, edgecolor='black', linewidth=1, alpha=0.5, hatch='//')

# Value labels — above bars with enough clearance
for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
           f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
           f'{val:.1f}', ha='center', fontsize=9)

# Inductive bias labels — BELOW x-axis method names, as colored text
bias_labels = ['Sequential\n+ pretrained', 'Sequential\n(no pretrain)', 'Permutation-\ninvariant', 'Permutation-\ninvariant']
for i, (bias, c) in enumerate(zip(bias_labels, colors)):
    ax.text(i, -8, bias, ha='center', fontsize=7.5, color=c, fontweight='bold', linespacing=1.3)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Score')
ax.set_ylim(-12, 118)
ax.set_title('B. Structural Baseline Comparison', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=9, loc='center right', framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y')

# Annotation arrows — repositioned to avoid overlap
ax.annotate('Pretraining\nmismatch', xy=(0, 52), xytext=(0, 100),
           fontsize=7.5, ha='center', color='#B71C1C',
           arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.2),
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
ax.annotate('Sequential\nbias (−0.6pp)', xy=(1, 92), xytext=(1.15, 111),
           fontsize=7.5, ha='center', color='#FF9800',
           arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.2),
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
ax.annotate('Correct\ninductive bias', xy=(2.3, 92), xytext=(2.85, 100),
           fontsize=7.5, ha='center', color='#4CAF50',
           arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2),
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig(f'{OUT}/inductive_bias_figure.png', dpi=300)
print(f"Saved: {OUT}/inductive_bias_figure.png")
