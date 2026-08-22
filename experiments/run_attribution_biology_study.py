#!/usr/bin/env python3
"""
Exp A + B: LOO Attribution Biology + Study Effect Analysis
============================================================
Exp A: LOO attribution vs differential abundance Spearman correlation
       (parallel to BiomeGPT's attention vs |log2 fold-change| analysis)
Exp B: Does embedding cluster by study source or by disease?
       (parallel to MGM's batch integration analysis)

Exp E: Embedding similarity heatmap (top-30 genera)
"""
import json, os, sys, pickle, csv
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP A+B+E: Attribution Biology + Study Effect + Heatmap")
print("=" * 60)

# ── Load data ──
train_data = [json.loads(l) for l in open(f'{DATA_DIR}/train_nl.jsonl')]
test_data = [json.loads(l) for l in open(f'{DATA_DIR}/test_nl.jsonl')]
all_data = train_data + test_data
ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')
all_seqs = np.concatenate([ts, xs])
all_masks = np.concatenate([tm, xm])
all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in all_data])

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    info = json.load(f)
    GENUS_NAMES = info['genus_names']
    sources = info.get('sources', [])

# Map sample to source
sample_sources = {}
for i, sid in enumerate(info.get('sample_ids', [])):
    if i < len(sources):
        sample_sources[sid] = sources[i]

# ── Load LOO attribution data ──
with open(f'{OUT_DIR}/shap_data_full.pkl', 'rb') as f:
    shap_by_id = {}
    for s in pickle.load(f)['all_samples']:
        shap_by_id[s['sample_id']] = {'label': s['label'], 'importance': s['importance']}

# ── Load embeddings ──
emb = np.load(f'{OUT_DIR}/../backbone/embeddings.npy')
sample_ids_all = np.load(f'{OUT_DIR}/../backbone/sample_ids.npy')

# ═══════════════════════════════════════════
# EXP A: LOO Attribution vs Differential Abundance
# ═══════════════════════════════════════════
print("\n[EXP A] LOO Attribution vs Differential Abundance")

# Compute per-genus differential abundance: mean abundance in Disease vs Healthy
V = 1226
# Build abundance matrix (presence count per genus per sample)
abundance_matrix = np.zeros((len(all_data), V), dtype=np.float32)
for i in range(len(all_data)):
    valid = all_masks[i].astype(bool)
    for j in range(len(all_seqs[i])):
        if valid[j] and all_seqs[i][j] > 0:
            gid = int(all_seqs[i][j])
            abundance_matrix[i, gid] += 1.0
    total = abundance_matrix[i].sum()
    if total > 0:
        abundance_matrix[i] /= total

# Differential abundance: log2 fold change (disease vs healthy)
eps = 1e-6
mean_disease = abundance_matrix[all_labels == 1].mean(axis=0)
mean_healthy = abundance_matrix[all_labels == 0].mean(axis=0)
log2fc = np.log2((mean_disease + eps) / (mean_healthy + eps))

# Global LOO importance per genus (from shap_data_full)
global_loo = {}
for s in shap_by_id.values():
    for g in s['importance']:
        gn = g['genus_name']
        if gn not in global_loo:
            global_loo[gn] = []
        global_loo[gn].append(abs(g['importance']))

# Match genera between LOO importance and abundance data
genus_ids_in_data = {}
for gn in global_loo:
    try:
        gid = GENUS_NAMES.index(gn) + 1
        genus_ids_in_data[gn] = gid
    except ValueError:
        pass

# Compute Spearman correlation between |log2FC| and mean LOO importance
print(f"  Genera in both analyses: {len(genus_ids_in_data)}")

# By prevalence strata
all_corr_results = {}
for min_prev, label in [(0, 'all'), (10, 'prev>=10'), (50, 'prev>=50'), (200, 'prev>=200')]:
    valid_genera = []
    for gn, gid in genus_ids_in_data.items():
        n_samples = abundance_matrix[:, gid].astype(bool).sum()
        if n_samples >= min_prev:
            valid_genera.append((gn, gid))

    if len(valid_genera) < 10:
        continue

    lfc_vals = [abs(log2fc[gid]) for gn, gid in valid_genera]
    loo_vals = [np.mean(global_loo[gn]) for gn, gid in valid_genera]
    rho, p = spearmanr(lfc_vals, loo_vals)
    all_corr_results[label] = {'rho': float(rho), 'p': float(p), 'n': len(valid_genera)}
    print(f"  {label}: |log2FC| vs LOO importance Spearman ρ={rho:.4f} (p={p:.4f}, n={len(valid_genera)})")

# Compare with BiomeGPT's results: they found ρ >= 0.40 for some tasks,
# weak/near-zero for others. Our result shows whether our attribution
# captures univariate differential abundance or higher-order structure.

# ═══════════════════════════════════════════
# EXP B: Study Effect Analysis
# ═══════════════════════════════════════════
print("\n[EXP B] Study Effect Analysis: embedding clusters by disease or study?")

# Group samples by source
source_groups = {}
for i, sid in enumerate(sample_ids_all):
    src = str(sample_sources.get(str(sid), 'unknown'))
    if src not in source_groups:
        source_groups[src] = []
    source_groups[src].append(i)

print(f"  Source groups: {[(k, len(v)) for k, v in sorted(source_groups.items(), key=lambda x: -len(x[1]))]}")

# Compute: intra-source cosine similarity vs inter-source cosine similarity
emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

# Disease vs Healthy separation within same source
# (if sources are mixed, compare disease-structure vs source-structure)
intra_source_sims = []
for src, indices in source_groups.items():
    if len(indices) < 10:
        continue
    idx = np.array(indices)
    sub = emb_norm[idx]
    sim = sub @ sub.T
    n = len(idx)
    mask = ~np.eye(n, dtype=bool)
    intra_source_sims.append(sim[mask].mean())

# Disease structure: cosine between same-label samples
disease_idx = np.where(all_labels == 1)[0]
healthy_idx = np.where(all_labels == 0)[0]
d_sim = emb_norm[disease_idx] @ emb_norm[disease_idx].T
h_sim = emb_norm[healthy_idx] @ emb_norm[healthy_idx].T
n_d = len(disease_idx); n_h = len(healthy_idx)
intra_disease = d_sim[~np.eye(n_d, dtype=bool)].mean()
intra_healthy = h_sim[~np.eye(n_h, dtype=bool)].mean()

print(f"  Intra-source cosine similarity: {np.mean(intra_source_sims):.4f}")
print(f"  Intra-disease cosine: {intra_disease:.4f}")
print(f"  Intra-healthy cosine: {intra_healthy:.4f}")

# Key question: does embedding separate by disease WITHIN a source?
for src, indices in source_groups.items():
    if len(indices) < 20:
        continue
    idx = np.array(indices)
    sub_emb = emb_norm[idx]
    sub_labels = all_labels[idx]
    d_mask = sub_labels == 1
    h_mask = sub_labels == 0
    if d_mask.sum() < 5 or h_mask.sum() < 5:
        continue
    d_sim_in = sub_emb[d_mask] @ sub_emb[d_mask].T
    h_sim_in = sub_emb[h_mask] @ sub_emb[h_mask].T
    cross_sim = sub_emb[d_mask] @ sub_emb[h_mask].T
    nd = d_mask.sum(); nh = h_mask.sum()
    intra_d = d_sim_in[~np.eye(nd, dtype=bool)].mean() if nd > 1 else 0
    intra_h = h_sim_in[~np.eye(nh, dtype=bool)].mean() if nh > 1 else 0
    cross = cross_sim.mean()
    print(f"  {src} (n={len(idx)}): D-D={intra_d:.3f} H-H={intra_h:.3f} D-H={cross:.3f}")

# ═══════════════════════════════════════════
# EXP E: Embedding Similarity Heatmap (top-30 genera)
# ═══════════════════════════════════════════
print("\n[EXP E] Embedding Similarity Heatmap")

import torch, torch.nn as nn
# Load trained model to get genus embeddings
class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1226, 768, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

model_state = torch.load(f'{OUT_DIR}/../backbone/final_model.pt', map_location='cpu')
# Extract embedding weight
emb_weight = None
for key, val in model_state.items():
    if 'emb.weight' in key:
        emb_weight = val.numpy()
        break

if emb_weight is None:
    # fallback: train quickly
    print("  No embedding weight in checkpoint, using random init...")
    emb_weight = np.random.randn(1226, 768).astype(np.float32)
    emb_weight[0] = 0

print(f"  Embedding weight shape: {emb_weight.shape}")

# Select top-30 genera by LOO importance
top_genera_sorted = sorted(global_loo.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)
top30_names = [gn for gn, _ in top_genera_sorted[:30]]
top30_ids = [GENUS_NAMES.index(gn) + 1 for gn in top30_names if gn in GENUS_NAMES]

# Compute cosine similarity matrix
top30_emb = emb_weight[top30_ids]
top30_norm = top30_emb / (np.linalg.norm(top30_emb, axis=1, keepdims=True) + 1e-8)
sim_matrix = top30_norm @ top30_norm.T

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-0.5, vmax=1.0)
ax.set_xticks(range(len(top30_names)))
ax.set_yticks(range(len(top30_names)))
ax.set_xticklabels([n[:12] for n in top30_names], rotation=90, fontsize=7)
ax.set_yticklabels([n[:12] for n in top30_names], fontsize=7)
ax.set_title('Genus Embedding Cosine Similarity (Top-30 LOO-important genera)', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/embedding_heatmap.png', dpi=200, bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/embedding_heatmap.png")

# Save results
results = {
    'exp_a_attribution_vs_foldchange': all_corr_results,
    'exp_b_study_effect': {
        'intra_source_cosine': float(np.mean(intra_source_sims)),
        'intra_disease_cosine': float(intra_disease),
        'intra_healthy_cosine': float(intra_healthy),
        'n_sources': len([s for s in source_groups if len(source_groups[s]) >= 10]),
    },
    'exp_e_top30_genera': top30_names,
}
with open(f'{OUT_DIR}/attribution_biology_study.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/attribution_biology_study.json")
print("\nDONE")
