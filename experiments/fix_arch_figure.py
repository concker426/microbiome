#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix Figure 1 (dataset_architecture_figure.png) layout."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _p in [
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf',
]:
    try:
        font_manager.fontManager.addfont(_p)
    except Exception:
        pass

from matplotlib.patches import FancyBboxPatch

OUT = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
plt.rcParams.update({'font.size': 12, 'font.family': 'Liberation Serif'})

fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 28)
ax.set_ylim(0, 11)
ax.axis('off')

def container(ax, x0, y0, w, h, edgecolor, facecolor, lw=1.6):
    r = FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.15',
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=lw, alpha=0.30)
    ax.add_patch(r)

def box(ax, x, y, w, h, text, color, fs=11, bold=False):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.2',
                       facecolor=color, edgecolor='black',
                       linewidth=1.3, alpha=0.95)
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=('bold' if bold else 'normal'))

def arrow(ax, x1, y1, x2, y2, lw=1.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color='#333333'))

# background containers
container(ax, 0.6, 0.4, 12.2, 10.2, '#B71C1C', '#FFEBEE')
container(ax, 13.6, 0.4, 13.6, 10.2, '#1B5E20', '#E8F5E9')

# MGM (left)
ax.text(6.7, 10.15, 'MGM (pretrained foundation model)', fontsize=13,
        fontweight='bold', ha='center', color='#B71C1C')
box(ax, 6.7, 9.05, 5.2, 0.8, 'Genus abundance profile', '#FFCDD2', 11)
arrow(ax, 6.7, 8.65, 6.7, 8.25)
box(ax, 6.7, 7.8, 5.2, 0.8, 'Token embedding (768-d)', '#FFCDD2', 11)
arrow(ax, 6.7, 7.4, 6.7, 7.0)
box(ax, 6.7, 6.55, 5.2, 0.8, '8-layer Transformer (34M)', '#EF9A9A', 11)
arrow(ax, 6.7, 6.15, 6.7, 5.75)
box(ax, 6.7, 5.3, 5.2, 0.8, 'Attention pooling', '#EF9A9A', 11)
arrow(ax, 6.7, 4.9, 6.7, 4.5)
box(ax, 6.7, 4.05, 5.2, 0.8, 'MLP classifier', '#E57373', 11)
arrow(ax, 6.7, 3.65, 6.7, 3.25)
box(ax, 6.7, 2.8, 5.2, 0.8, '"IBD" / "Healthy"', '#B71C1C', 11, bold=True)
ax.text(6.7, 1.15,
        '34M params  ·  pretrained on 263k samples' + chr(10) + '50.9% ACC (below majority)',
        fontsize=10.5, ha='center', va='center', color='#B71C1C', style='italic')

# SimpleEmb (right)
ax.text(20.4, 10.15, 'SimpleEmb (ours)', fontsize=13, fontweight='bold',
        ha='center', color='#1B5E20')
box(ax, 20.4, 9.05, 5.2, 0.8, 'Genus abundance profile', '#C8E6C9', 11)
arrow(ax, 20.4, 8.65, 20.4, 8.25)
box(ax, 20.4, 7.8, 5.2, 0.8, 'nn.Embedding(1226, 512)', '#A5D6A7', 11)
arrow(ax, 20.4, 7.4, 20.4, 7.0)
box(ax, 20.4, 6.55, 5.2, 0.8, 'Masked mean pooling  (h, 512-d)', '#A5D6A7', 11)
arrow(ax, 20.4, 6.15, 20.4, 5.75)
box(ax, 20.4, 5.25, 1.6, 0.65, 'h', '#81C784', 12, bold=True)
arrow(ax, 19.6, 4.95, 17.3, 4.15)
arrow(ax, 21.2, 4.95, 23.6, 4.15)
box(ax, 17.3, 3.6, 4.0, 0.9, 'MLP classifier' + chr(10) + '(512-256-2)', '#66BB6A', 10.5)
box(ax, 23.6, 3.6, 4.4, 0.9, 'LOO attribution' + chr(10) + '+ Qwen2-7B explanation', '#4CAF50', 10)
arrow(ax, 17.3, 3.1, 17.3, 2.65)
arrow(ax, 23.6, 3.1, 23.6, 2.65)
box(ax, 17.3, 2.15, 3.6, 0.8, '"IBD" / "Healthy"', '#1B5E20', 11, bold=True)
box(ax, 23.6, 2.15, 4.0, 0.8, 'Per-genus evidence', '#1B5E20', 11, bold=True)
ax.text(20.4, 1.15,
        '1.1M params  ·  no pretraining  ·  92.57% ACC' + chr(10) + 'LOO attribution + LLM interpretation',
        fontsize=10.5, ha='center', va='center', color='#1B5E20', style='italic')

fig.tight_layout(pad=0.6)
fig.savefig(OUT + '/dataset_architecture_figure.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved -> ' + OUT + '/dataset_architecture_figure.png')