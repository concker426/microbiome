# -*- coding: utf-8 -*-
"""Publication-quality matplotlib style for the SimpleEmb paper (ICLR 2027).
Body text is Times (iclr2027_conference + times); figures use Liberation Serif
(metric-compatible with Times New Roman) so the two match.
Applied via: exec(open('/hd/liujx/microbiome_llm_project/experiments/fig_style.py').read())
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

for path in [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
]:
    try:
        font_manager.fontManager.addfont(path)
    except Exception:
        pass

plt.rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.prop_cycle": plt.cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#8C8C8C"]),
})

OUR_COLOR = "#D55E00"
BASELINE_COLOR = "#B0BEC5"
