#!/usr/bin/env python3
"""Generate inductive bias illustration figure for Section 4.5"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── Panel A: Inductive Bias Comparison ──
ax = axes[0]
ax.set_xlim(0, 20); ax.set_ylim(0, 16); ax.axis('off')
ax.set_title('A. Inductive Bias Comparison', fontsize=13, fontweight='bold', loc='left')

# Sequential (Transformer) side
ax.text(5, 15.5, 'Sequential (Transformer)', fontsize=12, fontweight='bold', ha='center', color='#B71C1C')
# Draw "sequence" of genus tokens
for i in range(6):
    x = 2 + i*2.2
    y = 14
    rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.2",
                          facecolor='#FFCDD2', edgecolor='#B71C1C', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, f'g{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
# Arrows between tokens (position matters)
for i in range(5):
    ax.annotate('', xy=(3.4+i*2.2, 14), xytext=(2.4+i*2.2, 14),
               arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2))
ax.text(8, 13.2, 'Position matters: g1→g2→g3...', ha='center', fontsize=9, color='#B71C1C', style='italic')
ax.text(8, 12.7, 'Self-attention over ordered sequence', ha='center', fontsize=9, color='#B71C1C', style='italic')

# Arrow down
ax.annotate('', xy=(8, 11.5), xytext=(8, 12.2),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(8, 11.2, 'FT-Transformer: 91.0% ACC', ha='center', fontsize=10, color='#B71C1C')
ax.text(8, 10.7, 'MGM (pretrained): 50.9% ACC', ha='center', fontsize=10, color='#B71C1C')

# Permutation-invariant (Set) side
ax.text(15, 15.5, 'Permutation-Invariant (Set)', fontsize=12, fontweight='bold', ha='center', color='#1B5E20')
# Draw set of unordered elements
centers = [(13, 13.2), (16, 12.5), (15, 11.8), (13.5, 11.2), (16.5, 13.8), (17.5, 12)]
for i, (cx, cy) in enumerate(centers):
    circle = plt.Circle((cx, cy), 0.6, facecolor='#C8E6C9', edgecolor='#1B5E20', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(cx, cy, f'g{i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

# No arrows between elements
ax.text(15, 10.6, 'Position irrelevant: {g1, g2, ..., gk}', ha='center', fontsize=9, color='#1B5E20', style='italic')
ax.text(15, 10.1, 'Aggregate over SET, not sequence', ha='center', fontsize=9, color='#1B5E20', style='italic')

# Arrow down
ax.annotate('', xy=(15, 9.2), xytext=(15, 9.9),
           arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(15, 8.9, 'DeepSets: 91.6% ACC', ha='center', fontsize=10, color='#1B5E20')
ax.text(15, 8.4, 'ProCyon v2: 91.6% ACC', ha='center', fontsize=10, color='#1B5E20', fontweight='bold')

# Plus annotation
ax.text(15, 7.7, '+ explicit embedding space → SHAP, kNN, clustering, LLM', ha='center',
       fontsize=9, color='#1565C0', style='italic')

# Key insight box
rect = FancyBboxPatch((2, 4), 16, 3, boxstyle="round,pad=0.5",
                      facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5, alpha=0.5)
ax.add_patch(rect)
ax.text(10, 6.2, 'Key Insight: Microbiome abundance profiles are SETS, not sequences.\n'
        'Permutation-invariant models (DeepSets, SimpleEmb) match this structure.\n'
        'Transformers can work (FT: 91.0%) but carry unnecessary sequential bias.',
        ha='center', fontsize=10, fontweight='bold', color='#1565C0')

# ── Panel B: Structural Baseline Results ──
ax = axes[1]
methods = ['MGM\n(pretrained)', 'FT-Transformer\n(no pretrain)', 'DeepSets\n(set)', 'ProCyon v2\n(set+embed)']
accs = [50.9, 91.0, 91.6, 91.6]
aucs = [46.3, 95.5, 97.3, 96.4]
biases = ['Sequential\n+ pretrained', 'Sequential\n(no pretrain)', 'Permutation-\ninvariant', 'Permutation-\ninvariant']
colors = ['#F44336', '#FF9800', '#4CAF50', '#1B5E20']

x = np.arange(len(methods)); w = 0.3
bars1 = ax.bar(x - w/2, accs, w, label='ACC (%)', color=colors, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + w/2, [a*100 for a in aucs], w, label='AUC (×100)',
              color=[c for c in colors], edgecolor='black', linewidth=1, alpha=0.5, hatch='//')

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%',
           ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val*100:.1f}',
           ha='center', fontsize=9)

# Add inductive bias labels
for i, bias in enumerate(biases):
    ax.text(i, -5, bias, ha='center', fontsize=8, color=colors[i], fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Score'); ax.set_ylim(0, 110)
ax.set_title('B. Structural Baseline Comparison', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')

# Annotation arrows
ax.annotate('Pretraining\nmismatch', xy=(0, 50.9), xytext=(0.5, 30),
           fontsize=8, ha='center', color='#B71C1C',
           arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))
ax.annotate('Sequential bias\n(-0.6pp)', xy=(1, 91.0), xytext=(0.6, 75),
           fontsize=8, ha='center', color='#FF9800',
           arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))
ax.annotate('Correct\ninductive bias', xy=(2, 91.6), xytext=(3, 85),
           fontsize=8, ha='center', color='#4CAF50',
           arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUT}/inductive_bias_figure.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT}/inductive_bias_figure.png")
