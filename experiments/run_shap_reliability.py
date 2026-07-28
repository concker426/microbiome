#!/usr/bin/env python3
"""
Exp 5: SHAP Reliability Analysis
=================================
1. Spearman rank correlation across folds (not just Jaccard)
2. Deletion test: remove top-K SHAP genera, re-predict, measure drop
3. Random baseline: compare real SHAP vs permuted-label SHAP
"""
import json, os, sys, pickle, csv
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

V = 1226; E = 768; BS = 32; LR_RATE = 1e-3; WD = 1e-4; NE = 50
DEVICE = 'cuda:0'
SEEDS = [42, 123, 456, 789, 1024]

print("=" * 60)
print("EXP 5: SHAP RELIABILITY ANALYSIS")
print("=" * 60)

# ── Load data ──
train_data = [json.loads(l) for l in open(f'{DATA_DIR}/train_nl.jsonl')]
test_data = [json.loads(l) for l in open(f'{DATA_DIR}/test_nl.jsonl')]
ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES = json.load(f)['genus_names']

# ── Model ──
class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids); mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return (x*mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

class MLPHead(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(E, hidden); self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SimpleEmbEnc(); self.mlp = MLPHead()
    def forward(self, ids, mask=None):
        return self.mlp(self.enc(ids, mask))

def build_ds(data, seqs, masks):
    class DS(Dataset):
        def __init__(sf):
            sf.seqs=seqs; sf.masks=masks
            sf.labels=np.array([1 if d['label']=='Disease' else 0 for d in data])
        def __len__(sf): return len(sf.labels)
        def __getitem__(sf, i):
            return (torch.tensor(sf.seqs[i].astype(np.int64),dtype=torch.long),
                    torch.tensor(sf.masks[i],dtype=torch.bool),
                    torch.tensor(sf.labels[i],dtype=torch.long))
    return DS()

def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]; y=torch.stack([x[2] for x in batch])
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y

# ═══════════════════════════════════════════
# 1. SPEARMAN RANK CORRELATION across folds
# ═══════════════════════════════════════════
print("\n[1] Spearman Rank Correlation across folds...")

# Train 5 fold models (seed=42, same data)
from sklearn.model_selection import StratifiedKFold
train_labels_arr = np.array([1 if d['label']=='Disease' else 0 for d in train_data])
train_ds = build_ds(train_data, ts, tm)

fold_importance_ranks = []
for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(train_labels_arr)), train_labels_arr)):
    torch.manual_seed(42); np.random.seed(42)
    model = Model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)

    fold_train = torch.utils.data.Subset(train_ds, train_idx)
    train_loader = DataLoader(fold_train, batch_size=BS, shuffle=True, collate_fn=collate)
    for ep in range(NE):
        model.train()
        for gi, gm, y in train_loader:
            gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(gi, gm), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    # Compute SHAP (LOO) on test set
    model.eval()
    test_loader = DataLoader(Subset(train_ds, val_idx), batch_size=1, shuffle=False, collate_fn=collate)
    genus_imps = {}
    with torch.no_grad():
        for gi, gm, y in test_loader:
            gi, gm = gi.to(DEVICE), gm.to(DEVICE)
            prob_full = F.softmax(model(gi, gm), dim=1)[0, 1].item()
            valid = gm[0].bool(); n_valid = valid.sum().item()
            if n_valid <= 1: continue
            emb = model.enc.emb(gi[0, valid])
            sum_all = emb.sum(dim=0)
            for j in range(n_valid):
                loo_mean = (sum_all - emb[j]) / (n_valid - 1)
                prob_loo = F.softmax(model.mlp(loo_mean.unsqueeze(0)), dim=1)[0, 1].item()
                gid = int(gi[0, valid][j].item())
                gname = GENUS_NAMES[gid-1] if gid-1 < len(GENUS_NAMES) else f'g{gid}'
                imp = prob_full - prob_loo
                if gname not in genus_imps: genus_imps[gname] = []
                genus_imps[gname].append(imp)

    # Aggregate: mean importance per genus
    mean_imps = {g: np.mean(v) for g, v in genus_imps.items()}
    fold_importance_ranks.append(mean_imps)
    del model; torch.cuda.empty_cache()
    print(f"  Fold {fold}: {len(mean_imps)} genera evaluated")

# Compute pairwise Spearman correlation
common_genera = set.intersection(*[set(d.keys()) for d in fold_importance_ranks])
spearman_results = []
for i in range(5):
    for j in range(i+1, 5):
        vals_i = [fold_importance_ranks[i][g] for g in common_genera]
        vals_j = [fold_importance_ranks[j][g] for g in common_genera]
        rho, p = spearmanr(vals_i, vals_j)
        spearman_results.append({'fold_pair': f'{i}-{j}', 'rho': float(rho), 'p': float(p)})

mean_rho = np.mean([s['rho'] for s in spearman_results])
print(f"  Mean Spearman ρ across folds: {mean_rho:.4f}")
for s in spearman_results:
    print(f"    Fold {s['fold_pair']}: ρ={s['rho']:.4f} p={s['p']:.4f}")

# ═══════════════════════════════════════════
# 2. DELETION TEST
# ═══════════════════════════════════════════
print("\n[2] Deletion Test: removing top-K SHAP genera...")

# Train full model on all train data
torch.manual_seed(42); np.random.seed(42)
model = Model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
full_train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
for ep in range(NE):
    model.train()
    for gi, gm, y in full_train_loader:
        gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(gi, gm), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

# Load global SHAP ranking
with open(f'{OUT_DIR}/shap_data_full.pkl', 'rb') as f:
    shap_data = pickle.load(f)

# Compute global mean importance
global_imps = {}
for s in shap_data['all_samples']:
    for g in s['importance']:
        gn = g['genus_name']
        if gn not in global_imps: global_imps[gn] = []
        global_imps[gn].append(g['importance'])

global_ranked = sorted(global_imps.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)
top_genera = [g for g, _ in global_ranked[:50]]

# Deletion experiment
test_ds = build_ds(test_data, xs, xm)
test_loader_full = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)

model.eval()
# Baseline (no deletion)
all_probs_full = []
with torch.no_grad():
    for gi, gm, y in test_loader_full:
        gi, gm = gi.to(DEVICE), gm.to(DEVICE)
        all_probs_full.append(F.softmax(model(gi, gm), dim=1)[:, 1].cpu().numpy())
probs_full = np.concatenate(all_probs_full)
test_labels_arr = np.array([1 if d['label']=='Disease' else 0 for d in test_data])
baseline_auc = float(roc_auc_score(test_labels_arr, probs_full))

# Delete top-K genera and re-evaluate
deletion_results = {}
for k in [5, 10, 20, 50]:
    # Create modified test data with top-K genera removed (set to padding 0)
    xs_modified = xs.copy()
    xm_modified = xm.copy()

    # Find which genus IDs correspond to top-K genera
    top_k_names = [g for g, _ in global_ranked[:k]]
    top_k_ids = set()
    for gname in top_k_names:
        try:
            gid = GENUS_NAMES.index(gname) + 1
            top_k_ids.add(gid)
        except ValueError:
            pass

    # Zero out top-K genera in test data
    for i in range(len(xs_modified)):
        for pos in range(len(xs_modified[i])):
            if xm_modified[i, pos] and xs_modified[i, pos] in top_k_ids:
                xs_modified[i, pos] = 0
                xm_modified[i, pos] = 0

    test_ds_mod = build_ds(test_data, xs_modified, xm_modified)
    test_loader_mod = DataLoader(test_ds_mod, batch_size=BS, shuffle=False, collate_fn=collate)

    all_probs_mod = []
    with torch.no_grad():
        for gi, gm, y in test_loader_mod:
            gi, gm = gi.to(DEVICE), gm.to(DEVICE)
            all_probs_mod.append(F.softmax(model(gi, gm), dim=1)[:, 1].cpu().numpy())
    probs_mod = np.concatenate(all_probs_mod)
    auc_mod = float(roc_auc_score(test_labels_arr, probs_mod))
    acc_full = accuracy_score(test_labels_arr, probs_full > 0.5)
    acc_mod = accuracy_score(test_labels_arr, probs_mod > 0.5)

    deletion_results[f'K={k}'] = {
        'baseline_auc': baseline_auc, 'deleted_auc': auc_mod,
        'auc_drop': baseline_auc - auc_mod,
        'baseline_acc': float(acc_full), 'deleted_acc': float(acc_mod),
        'acc_drop': float(acc_full - acc_mod),
    }
    print(f"  Delete top-{k}: AUC {baseline_auc:.4f}→{auc_mod:.4f} (drop={baseline_auc-auc_mod:.4f}) "
          f"ACC {acc_full:.4f}→{acc_mod:.4f}")

# Random deletion baseline (delete K random genera)
rng = np.random.RandomState(42)
random_drop_results = {}
for k in [5, 10, 20, 50]:
    all_aucs = []
    for trial in range(10):
        random_ids = set(rng.choice(range(1, V), k, replace=False))
        xs_rand = xs.copy(); xm_rand = xm.copy()
        for i in range(len(xs_rand)):
            for pos in range(len(xs_rand[i])):
                if xm_rand[i, pos] and xs_rand[i, pos] in random_ids:
                    xs_rand[i, pos] = 0; xm_rand[i, pos] = 0
        test_ds_rand = build_ds(test_data, xs_rand, xm_rand)
        test_loader_rand = DataLoader(test_ds_rand, batch_size=BS, shuffle=False, collate_fn=collate)
        all_probs_rand = []
        with torch.no_grad():
            for gi, gm, y in test_loader_rand:
                gi, gm = gi.to(DEVICE), gm.to(DEVICE)
                all_probs_rand.append(F.softmax(model(gi, gm), dim=1)[:, 1].cpu().numpy())
        probs_rand = np.concatenate(all_probs_rand)
        all_aucs.append(float(roc_auc_score(test_labels_arr, probs_rand)))
    random_drop_results[f'K={k}'] = {'mean_auc': float(np.mean(all_aucs)), 'std_auc': float(np.std(all_aucs))}
    print(f"  Random delete {k} (10 trials): AUC={np.mean(all_aucs):.4f}±{np.std(all_aucs):.4f}")

# ═══════════════════════════════════════════
# 3. RANDOM BASELINE (permutation control)
# ═══════════════════════════════════════════
print("\n[3] Permutation Control: model trained on shuffled labels...")

torch.manual_seed(42); np.random.seed(42)
shuffled_labels = train_labels_arr.copy()
np.random.shuffle(shuffled_labels)

class ShuffledDS(Dataset):
    def __init__(self):
        self.seqs=ts; self.masks=tm; self.labels=shuffled_labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                torch.tensor(self.masks[i],dtype=torch.bool),
                torch.tensor(self.labels[i],dtype=torch.long))

shuffled_ds = ShuffledDS()
shuffled_loader = DataLoader(shuffled_ds, batch_size=BS, shuffle=True, collate_fn=collate)

perm_model = Model().to(DEVICE)
opt = torch.optim.AdamW(perm_model.parameters(), lr=LR_RATE, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
for ep in range(NE):
    perm_model.train()
    for gi, gm, y in shuffled_loader:
        gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(perm_model(gi, gm), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

# Compute SHAP on test set for permuted model
perm_model.eval()
test_loader_single = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)
perm_shap_values = []
with torch.no_grad():
    for gi, gm, y in test_loader_single:
        gi, gm = gi.to(DEVICE), gm.to(DEVICE)
        prob_full = F.softmax(perm_model(gi, gm), dim=1)[0, 1].item()
        valid = gm[0].bool(); n_valid = valid.sum().item()
        if n_valid <= 1: continue
        emb = perm_model.enc.emb(gi[0, valid])
        sum_all = emb.sum(dim=0)
        for j in range(min(n_valid, 10)):  # sample 10 per patient for speed
            loo_mean = (sum_all - emb[j]) / (n_valid - 1)
            prob_loo = F.softmax(perm_model.mlp(loo_mean.unsqueeze(0)), dim=1)[0, 1].item()
            perm_shap_values.append(abs(prob_full - prob_loo))

# Compare with real SHAP values
real_shap_values = []
for s in shap_data['all_samples']:
    for g in s['importance']:
        real_shap_values.append(abs(g['importance']))

real_mean = float(np.mean(real_shap_values))
perm_mean = float(np.mean(perm_shap_values))
ratio = real_mean / max(perm_mean, 1e-10)
print(f"  Real SHAP mean: {real_mean:.6f}")
print(f"  Permuted SHAP mean: {perm_mean:.6f}")
print(f"  Signal ratio: {ratio:.1f}x")

# ═══════════════════════════════════════════
# 4. VISUALIZATION
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# A: Spearman correlation matrix
ax = axes[0]
rho_matrix = np.ones((5, 5))
for s in spearman_results:
    i, j = map(int, s['fold_pair'].split('-'))
    rho_matrix[i, j] = s['rho']; rho_matrix[j, i] = s['rho']
im = ax.imshow(rho_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f'F{i}' for i in range(5)]); ax.set_yticklabels([f'F{i}' for i in range(5)])
ax.set_title(f'A. SHAP Rank Correlation\n(Mean ρ={mean_rho:.3f})', fontweight='bold', loc='left')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{rho_matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# B: Deletion test
ax = axes[1]
ks = [5, 10, 20, 50]
drops = [deletion_results[f'K={k}']['auc_drop'] for k in ks]
rand_drops = [baseline_auc - random_drop_results[f'K={k}']['mean_auc'] for k in ks]
ax.plot(ks, drops, 'o-', label='SHAP top-K deletion', color='#F44336', markersize=10, linewidth=2)
ax.plot(ks, rand_drops, 's--', label='Random deletion (10 trials)', color='#9E9E9E', markersize=8)
ax.set_xlabel('K genera removed'); ax.set_ylabel('AUC Drop')
ax.set_title('B. Deletion Test', fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# C: SHAP distribution: Real vs Permuted
ax = axes[2]
bins = np.linspace(0, np.percentile(real_shap_values, 99), 40)
ax.hist(real_shap_values, bins=bins, alpha=0.7, label=f'Real (μ={real_mean:.5f})', color='#4CAF50')
ax.hist(perm_shap_values, bins=bins, alpha=0.7, label=f'Permuted (μ={perm_mean:.5f})', color='#F44336')
ax.set_xlabel('|SHAP Importance|'); ax.set_ylabel('Frequency')
ax.set_title(f'C. Signal Quality ({ratio:.0f}×)', fontweight='bold', loc='left')
ax.legend(fontsize=8)

# D: Prevalence vs SHAP rank stability
ax = axes[3]
with open(f'{OUT_DIR}/global_importance_full.csv') as f:
    global_data = list(csv.DictReader(f))
prevalence = [int(r['n_samples']) for r in global_data]
importance = [abs(float(r['mean_importance'])) for r in global_data]
ax.scatter(prevalence, importance, s=30, alpha=0.4, c='#1565C0', edgecolors='none')
ax.set_xlabel('Prevalence (n samples)'); ax.set_ylabel('|SHAP Importance|')
r, p = pearsonr(prevalence, importance)
ax.set_title(f'D. Prevalence vs Importance\n(r={r:.3f}, p={p:.4f})', fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_reliability.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR}/shap_reliability.png")

# ═══════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════
results = {
    'spearman_correlation': {'mean_rho': float(mean_rho), 'pairwise': spearman_results},
    'deletion_test': deletion_results,
    'random_deletion': random_drop_results,
    'permutation_control': {'real_mean': real_mean, 'permuted_mean': perm_mean, 'ratio': float(ratio)},
    'prevalence_correlation': {'pearson_r': float(r), 'pearson_p': float(p)},
}
with open(f'{OUT_DIR}/shap_reliability.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/shap_reliability.json")
print("EXP 5 DONE")
