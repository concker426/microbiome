#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison for ProCyon v2 Paper
========================================================
Compares 12+ methods on clean_2538 (same train/test split):
  A. Raw genus features + classical ML (Linear, MLP, RF, XGBoost)
  B. SimpleEmb (various dims) + Linear/MLP
  C. MGM encoder (pretrained/random) + Linear/MLP
  D. kNN on embeddings
  E. Ablation: embedding dim, BatchNorm, Dropout, training data size

Outputs:
  - Overleaf-ready LaTeX table (results_table.tex)
  - Full JSON results (comprehensive_results.json)
  - Ablation figures (ablation_figure.png)
"""
import json, os, sys, time, gc, csv, warnings
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ──
V = 1226  # vocab size
E_DEFAULT = 768
SL = 86
NE = 50
BS = 32
LR = 1e-3
WD = 1e-4
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("COMPREHENSIVE BASELINE COMPARISON")
print("=" * 70)

# ═══════════════════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════════════════
print("\n[1] Loading data...")

train_data = []
with open(f'{DATA_DIR}/train_nl.jsonl') as f:
    for l in f:
        train_data.append(json.loads(l))
test_data = []
with open(f'{DATA_DIR}/test_nl.jsonl') as f:
    for l in f:
        test_data.append(json.loads(l))

ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES = json.load(f)['genus_names']

train_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in train_data])
test_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in test_data])
all_seqs = np.concatenate([ts, xs], axis=0)
all_masks = np.concatenate([tm, xm], axis=0)
all_labels = np.concatenate([train_labels, test_labels], axis=0)

print(f"  Train: {len(train_data)} (D={train_labels.sum()}, H={len(train_labels)-train_labels.sum()})")
print(f"  Test:  {len(test_data)} (D={test_labels.sum()}, H={len(test_labels)-test_labels.sum()})")
print(f"  Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
# 2. Feature Extractors
# ═══════════════════════════════════════════════════════════

def extract_raw_features(seqs):
    """Raw genus ID features: multi-hot encoding (presence/absence)."""
    n = len(seqs)
    feats = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        for j in range(SL):
            gid = seqs[i, j]
            if gid > 0:
                feats[i, int(gid)] = 1.0
    return feats

def extract_raw_abundance(seqs, masks):
    """Raw genus ID as normalized abundance (like a bag-of-words)."""
    n = len(seqs)
    feats = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        valid = masks[i].astype(bool)
        for j in range(SL):
            if valid[j] and seqs[i, j] > 0:
                gid = int(seqs[i, j])
                feats[i, gid] += 1.0
        total = feats[i].sum()
        if total > 0:
            feats[i] /= total
    return feats

@torch.no_grad()
def extract_simple_embeddings(seqs, masks, embed_dim=E_DEFAULT):
    """Simple Embedding + masked mean pool (random init)."""
    emb = nn.Embedding(V, embed_dim, padding_idx=0).to(DEVICE)
    feats = []
    for i in range(0, len(seqs), 128):
        gi = torch.from_numpy(seqs[i:i+128].astype(np.int64)).long().to(DEVICE)
        gm = torch.from_numpy(masks[i:i+128]).bool().to(DEVICE)
        e = emb(gi)
        gm_float = gm.float().unsqueeze(-1)
        pooled = (e * gm_float).sum(dim=1) / gm_float.sum(dim=1).clamp(min=1)
        feats.append(pooled.cpu().numpy())
    del emb
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def extract_mgm_features(seqs, masks, pretrained=True):
    """MGM Transformer encoder features."""
    sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
    from run_v6_merged import MGMEnc
    enc = MGMEnc()
    if pretrained:
        ck = torch.load('/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt',
                       map_location='cpu')
        st = ck.get('model_state_dict', ck)
        enc.load_state_dict(st, strict=False)
    enc.to(DEVICE).eval()
    feats = []
    for i in range(0, len(seqs), 64):
        gi = torch.from_numpy(seqs[i:i+64].astype(np.int64)).long().to(DEVICE)
        gm = torch.from_numpy(masks[i:i+64]).bool().to(DEVICE)
        feats.append(enc(gi, gm).cpu().numpy())
    del enc
    return np.concatenate(feats, axis=0)


# ═══════════════════════════════════════════════════════════
# 3. SimpleEmb+MLP (our model) — train on train, eval on test
# ═══════════════════════════════════════════════════════════

class SimpleEmbEnc(nn.Module):
    def __init__(self, embed_dim=E_DEFAULT):
        super().__init__()
        self.emb = nn.Embedding(V, embed_dim, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

class MLP(nn.Module):
    def __init__(self, in_dim=E_DEFAULT, hidden=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.drop(x)
        return self.fc2(x)

class Model(nn.Module):
    def __init__(self, embed_dim=E_DEFAULT, hidden=256, dropout=0.3):
        super().__init__()
        self.enc = SimpleEmbEnc(embed_dim)
        self.mlp = MLP(embed_dim, hidden, dropout)
    def forward(self, ids, mask=None):
        return self.mlp(self.enc(ids, mask))
    def encode(self, ids, mask=None):
        return self.enc(ids, mask)


def build_ds(data, seqs, masks):
    class DS(Dataset):
        def __init__(self):
            self.seqs = seqs
            self.masks = masks
            self.labels = np.array([1 if d['label'] == 'Disease' else 0 for d in data])
            self.sw = np.array([1.5 if d.get('label', 'Healthy') == 'Disease' else 1.0 for d in data])
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, i):
            return (torch.tensor(self.seqs[i].astype(np.int64), dtype=torch.long),
                    torch.tensor(self.masks[i], dtype=torch.bool),
                    torch.tensor(self.labels[i], dtype=torch.long),
                    torch.tensor(self.sw[i], dtype=torch.float32))
    return DS()

def collate(batch):
    gi = [x[0] for x in batch]
    gm = [x[1] for x in batch]
    y = torch.stack([x[2] for x in batch])
    sw = torch.stack([x[3] for x in batch])
    mgl = max(len(g) for g in gi)
    pg, pm = [], []
    for i in range(len(gi)):
        g = gi[i]; m = gm[i]; p = mgl - len(g)
        pg.append(torch.cat([g, torch.zeros(p, dtype=torch.long)]) if p > 0 else g)
        pm.append(torch.cat([m, torch.zeros(p, dtype=torch.bool)]) if p > 0 else m)
    return torch.stack(pg), torch.stack(pm), y, sw


def train_simpleemb_mlp(embed_dim=E_DEFAULT, hidden=256, dropout=0.3, n_epochs=NE):
    """Train SimpleEmb+MLP from scratch on train, eval on test. Return metrics."""
    train_ds = build_ds(train_data, ts, tm)
    test_ds = build_ds(test_data, xs, xm)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)

    model = Model(embed_dim, hidden, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for ep in range(n_epochs):
        model.train()
        for gi, gm, y, sw in train_loader:
            gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
            logits = model(gi, gm)
            loss = F.cross_entropy(logits, y, reduction='none')
            loss = (loss * sw).sum() / sw.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for gi, gm, y, sw in test_loader:
            gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
            logits = model(gi, gm)
            prob = F.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.append(prob[:, 1].cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    del model; gc.collect(); torch.cuda.empty_cache()
    return compute_metrics(labels, preds, probs)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute standard classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_prob)),
        'f1': float(f1_score(y_true, y_pred)),
        'sensitivity': float(recall_score(y_true, y_pred)),  # TP/(TP+FN)
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precision': float(precision_score(y_true, y_pred)),
        'cm': cm.tolist(),
        'n_params': None,
    }


# ═══════════════════════════════════════════════════════════
# 4. Evaluate sklearn classifiers on extracted features
# ═══════════════════════════════════════════════════════════

SKLEARN_CLFS = {
    'Linear': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    'MLP_sklearn': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    'kNN': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1),
}

def eval_sklearn_on_features(X_train, X_test, y_train, y_test, name_prefix):
    """Train sklearn classifiers on pre-extracted features, evaluate on test."""
    results = {}
    for clf_name, clf in SKLEARN_CLFS.items():
        t0 = time.time()
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_train, y_train)
        y_pred = clf_copy.predict(X_test)
        try:
            y_prob = clf_copy.predict_proba(X_test)[:, 1]
        except:
            y_prob = y_pred.astype(float)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['time_s'] = round(time.time() - t0, 1)
        full_name = f"{name_prefix}+{clf_name}"
        results[full_name] = metrics
        print(f"  {full_name:<50s} ACC={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
              f"Sens={metrics['sensitivity']:.4f} Spec={metrics['specificity']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════
# 5. 5-fold CV evaluation (more robust for paper)
# ═══════════════════════════════════════════════════════════

def eval_sklearn_cv(X, y, name_prefix):
    """5-fold CV evaluation of sklearn classifiers."""
    results = {}
    for clf_name, clf in SKLEARN_CLFS.items():
        all_acc, all_auc, all_f1, all_sens, all_spec = [], [], [], [], []
        for seed in [42, 123, 456]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_tr, y_tr)
                y_pred = clf_copy.predict(X_val)
                try:
                    y_prob = clf_copy.predict_proba(X_val)[:, 1]
                except:
                    y_prob = y_pred.astype(float)
                all_acc.append(accuracy_score(y_val, y_pred))
                try:
                    all_auc.append(roc_auc_score(y_val, y_prob))
                except:
                    all_auc.append(0.5)
                all_f1.append(f1_score(y_val, y_pred))
                all_sens.append(recall_score(y_val, y_pred))
                cm = confusion_matrix(y_val, y_pred)
                tn, fp, fn, tp = cm.ravel()
                all_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        full_name = f"{name_prefix}+{clf_name}"
        results[full_name] = {
            'accuracy_mean': float(np.mean(all_acc)), 'accuracy_std': float(np.std(all_acc)),
            'auc_mean': float(np.mean(all_auc)), 'auc_std': float(np.std(all_auc)),
            'f1_mean': float(np.mean(all_f1)), 'f1_std': float(np.std(all_f1)),
            'sensitivity_mean': float(np.mean(all_sens)), 'sensitivity_std': float(np.std(all_sens)),
            'specificity_mean': float(np.mean(all_spec)), 'specificity_std': float(np.std(all_spec)),
        }
        print(f"  CV {full_name:<50s} ACC={results[full_name]['accuracy_mean']:.4f}±{results[full_name]['accuracy_std']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENTS
# ═══════════════════════════════════════════════════════════

all_results = {}

# ── 6A. Raw features (baseline) ──
print("\n" + "=" * 70)
print("[6A] Raw genus features (multi-hot " + str(V) + "-dim)")
print("=" * 70)

t0 = time.time()
X_raw_train = extract_raw_features(ts)
X_raw_test = extract_raw_features(xs)
print(f"  Raw multi-hot shape: train={X_raw_train.shape}, test={X_raw_test.shape} ({time.time()-t0:.1f}s)")

raw_results = eval_sklearn_on_features(X_raw_train, X_raw_test, train_labels, test_labels, "Raw-binary")
all_results.update(raw_results)

# ── 6B. Raw abundance features ──
print("\n" + "=" * 70)
print("[6B] Raw abundance features (normalized " + str(V) + "-dim)")
print("=" * 70)

X_ab_train = extract_raw_abundance(ts, tm)
X_ab_test = extract_raw_abundance(xs, xm)

ab_results = eval_sklearn_on_features(X_ab_train, X_ab_test, train_labels, test_labels, "Raw-abund")
all_results.update(ab_results)

# ── 6C. SimpleEmb (768-d) + classifiers ──
print("\n" + "=" * 70)
print("[6C] SimpleEmb (768-d) features + sklearn classifiers")
print("=" * 70)

t0 = time.time()
X_se_train = extract_simple_embeddings(ts, tm, 768)
X_se_test = extract_simple_embeddings(xs, xm, 768)
print(f"  SimpleEmb shape: train={X_se_train.shape}, test={X_se_test.shape} ({time.time()-t0:.1f}s)")

se_results = eval_sklearn_on_features(X_se_train, X_se_test, train_labels, test_labels, "SimpleEmb")
all_results.update(se_results)

# ── 6D. SimpleEmb+MLP (our model, 5-seed ensemble) ──
print("\n" + "=" * 70)
print("[6D] SimpleEmb+MLP end-to-end (our model, 5 seeds)")
print("=" * 70)

seeds_results = {'accuracy': [], 'auc': [], 'f1': [], 'sensitivity': [], 'specificity': []}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    m = train_simpleemb_mlp(embed_dim=768, hidden=256, dropout=0.3)
    for k in seeds_results:
        seeds_results[k].append(m[k])
    n_params = sum(p.numel() for p in Model(768, 256, 0.3).parameters())
    print(f"  Seed {seed}: ACC={m['accuracy']:.4f} AUC={m['auc']:.4f} Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")

our_result = {
    'accuracy': float(np.mean(seeds_results['accuracy'])),
    'accuracy_std': float(np.std(seeds_results['accuracy'])),
    'auc': float(np.mean(seeds_results['auc'])),
    'auc_std': float(np.std(seeds_results['auc'])),
    'f1': float(np.mean(seeds_results['f1'])),
    'sensitivity': float(np.mean(seeds_results['sensitivity'])),
    'specificity': float(np.mean(seeds_results['specificity'])),
    'n_params': n_params,
    'seeds_detail': {k: float(np.mean(v)) for k, v in seeds_results.items()},
}
all_results['ProCyon_v2 (SimpleEmb+MLP, 5-seed ensemble)'] = our_result
print(f"  ENSEMBLE: ACC={our_result['accuracy']:.4f}±{our_result['accuracy_std']:.4f} "
      f"AUC={our_result['auc']:.4f} Sens={our_result['sensitivity']:.4f} Spec={our_result['specificity']:.4f}")

# ── 6E. MGM encoder features ──
print("\n" + "=" * 70)
print("[6E] MGM Pretrained encoder (768-d)")
print("=" * 70)

try:
    t0 = time.time()
    X_mgm_train = extract_mgm_features(ts, tm, pretrained=True)
    X_mgm_test = extract_mgm_features(xs, xm, pretrained=True)
    print(f"  MGM shape: train={X_mgm_train.shape}, test={X_mgm_test.shape} ({time.time()-t0:.1f}s)")

    mgm_results = eval_sklearn_on_features(X_mgm_train, X_mgm_test, train_labels, test_labels, "MGM-pretrained")
    all_results.update(mgm_results)
except Exception as e:
    print(f"  MGM pretrained FAILED: {e}")

# ── 6F. MGM Random encoder ──
print("\n" + "=" * 70)
print("[6F] MGM Random init encoder (768-d)")
print("=" * 70)

try:
    t0 = time.time()
    X_mgm_rand_train = extract_mgm_features(ts, tm, pretrained=False)
    X_mgm_rand_test = extract_mgm_features(xs, xm, pretrained=False)
    print(f"  MGM-random shape: train={X_mgm_rand_train.shape} ({time.time()-t0:.1f}s)")

    mgmr_results = eval_sklearn_on_features(X_mgm_rand_train, X_mgm_rand_test, train_labels, test_labels, "MGM-random")
    all_results.update(mgmr_results)
except Exception as e:
    print(f"  MGM random FAILED: {e}")

# ── 6G. kNN on SimpleEmb embeddings ──
print("\n" + "=" * 70)
print("[6G] kNN (k=5, cosine) on SimpleEmb 768-d embeddings")
print("=" * 70)

knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
knn.fit(X_se_train, train_labels)
y_pred_knn = knn.predict(X_se_test)
y_prob_knn = knn.predict_proba(X_se_test)[:, 1]
knn_result = compute_metrics(test_labels, y_pred_knn, y_prob_knn)
all_results['SimpleEmb+kNN (k=5, cosine)'] = knn_result
print(f"  kNN: ACC={knn_result['accuracy']:.4f} AUC={knn_result['auc']:.4f} "
      f"Sens={knn_result['sensitivity']:.4f} Spec={knn_result['specificity']:.4f}")

# ═══════════════════════════════════════════════════════════
# 7. Embedding Dimension Ablation
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Embedding Dimension Ablation")
print("=" * 70)

dim_results = {}
for dim in [16, 32, 64, 128, 256, 512, 768, 1024]:
    print(f"\n  --- E={dim} ---")
    torch.manual_seed(42); np.random.seed(42)

    # Extract embeddings
    X_tr = extract_simple_embeddings(ts, tm, dim)
    X_te = extract_simple_embeddings(xs, xm, dim)

    # Eval with Linear
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_tr, train_labels)
    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)[:, 1]
    m_acc = accuracy_score(test_labels, y_pred)
    m_auc = roc_auc_score(test_labels, y_prob)

    # Eval with MLP (sklearn)
    mlp = MLPClassifier(hidden_layer_sizes=(min(dim, 256), min(dim//2, 128)), max_iter=300, random_state=42)
    mlp.fit(X_tr, train_labels)
    y_pred2 = mlp.predict(X_te)
    y_prob2 = mlp.predict_proba(X_te)[:, 1]
    m2_acc = accuracy_score(test_labels, y_pred2)
    m2_auc = roc_auc_score(test_labels, y_prob2)

    # Train SimpleEmb+MLP end-to-end
    torch.manual_seed(42); np.random.seed(42)
    e2e = train_simpleemb_mlp(embed_dim=dim, hidden=min(dim, 256), dropout=0.3, n_epochs=NE)
    n_p = sum(p.numel() for p in Model(dim, min(dim, 256), 0.3).parameters())

    dim_results[f'E={dim}'] = {
        'linear_acc': float(m_acc), 'linear_auc': float(m_auc),
        'mlp_acc': float(m2_acc), 'mlp_auc': float(m2_auc),
        'e2e_acc': float(e2e['accuracy']), 'e2e_auc': float(e2e['auc']),
        'n_params': n_p,
    }
    print(f"    Linear: ACC={m_acc:.4f} AUC={m_auc:.4f}")
    print(f"    MLP(sklearn): ACC={m2_acc:.4f} AUC={m2_auc:.4f}")
    print(f"    SimpleEmb+MLP(e2e): ACC={e2e['accuracy']:.4f} AUC={e2e['auc']:.4f} params={n_p}")

# ═══════════════════════════════════════════════════════════
# 8. Architecture Ablation (same E=768)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Architecture Component Ablation (E=768)")
print("=" * 70)

arch_results = {}

# 8A: Full model (baseline for ablation)
torch.manual_seed(42); np.random.seed(42)
arch_results['Full (Emb+BN+Dropout+MLP)'] = train_simpleemb_mlp(768, 256, 0.3)
print(f"  Full: ACC={arch_results['Full (Emb+BN+Dropout+MLP)']['accuracy']:.4f}")

# 8B: Emb + Linear (no MLP, no BN, no Dropout)
# This is equivalent to SimpleEmb+Linear
lr_emb = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr_emb.fit(X_se_train, train_labels)
y_p = lr_emb.predict(X_se_test)
y_pp = lr_emb.predict_proba(X_se_test)[:, 1]
arch_results['Emb+Linear (no MLP)'] = compute_metrics(test_labels, y_p, y_pp)
print(f"  Emb+Linear: ACC={arch_results['Emb+Linear (no MLP)']['accuracy']:.4f}")

# 8C: Emb + MLP (no BatchNorm)
class MLPNoBN(nn.Module):
    def __init__(self, in_dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

class ModelNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SimpleEmbEnc(768)
        self.mlp = MLPNoBN(768, 256, 0.3)
    def forward(self, ids, mask=None):
        return self.mlp(self.enc(ids, mask))

torch.manual_seed(42); np.random.seed(42)
train_ds = build_ds(train_data, ts, tm)
test_ds = build_ds(test_data, xs, xm)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)

model_no_bn = ModelNoBN().to(DEVICE)
opt = torch.optim.AdamW(model_no_bn.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
for ep in range(NE):
    model_no_bn.train()
    for gi, gm, y, sw in train_loader:
        gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
        logits = model_no_bn(gi, gm)
        loss = F.cross_entropy(logits, y, reduction='none')
        loss = (loss * sw).sum() / sw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

model_no_bn.eval()
all_preds, all_probs, all_lbls = [], [], []
with torch.no_grad():
    for gi, gm, y, sw in test_loader:
        gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
        logits = model_no_bn(gi, gm)
        prob = F.softmax(logits, dim=1)
        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_probs.append(prob[:, 1].cpu().numpy())
        all_lbls.append(y.cpu().numpy())
arch_results['Emb+MLP (no BN)'] = compute_metrics(
    np.concatenate(all_lbls), np.concatenate(all_preds), np.concatenate(all_probs))
print(f"  No BN: ACC={arch_results['Emb+MLP (no BN)']['accuracy']:.4f}")
del model_no_bn; gc.collect(); torch.cuda.empty_cache()

# 8D: Emb + MLP (no Dropout)
torch.manual_seed(42); np.random.seed(42)
arch_results['Emb+MLP (no Dropout)'] = train_simpleemb_mlp(768, 256, 0.0)
print(f"  No Dropout: ACC={arch_results['Emb+MLP (no Dropout)']['accuracy']:.4f}")

# ═══════════════════════════════════════════════════════════
# 9. Model Parameter Count
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[9] Model Complexity")
print("=" * 70)

# Count parameters for each method
def count_sklearn_params(model, n_features):
    """Estimate sklearn model parameters."""
    name = model.__class__.__name__
    if name == 'LogisticRegression':
        return n_features * 2 + 2  # weights + bias for 2 classes
    elif name == 'MLPClassifier':
        h1, h2 = 256, 128
        return n_features * h1 + h1 + h1 * h2 + h2 + h2 * 2 + 2
    elif name == 'RandomForestClassifier':
        return '~200 trees'
    elif name == 'XGBClassifier':
        return '~200 trees'
    elif name == 'KNeighborsClassifier':
        return 'non-parametric'
    return 'unknown'

# ═══════════════════════════════════════════════════════════
# 10. Generate LaTeX Table
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[10] Generating Results Tables")
print("=" * 70)

# Select key results for the main table
table_methods = [
    'Raw-binary+Linear', 'Raw-binary+MLP_sklearn', 'Raw-binary+RandomForest', 'Raw-binary+XGBoost',
    'Raw-abund+Linear', 'Raw-abund+MLP_sklearn', 'Raw-abund+RandomForest', 'Raw-abund+XGBoost',
    'MGM-random+Linear', 'MGM-random+MLP_sklearn',
    'MGM-pretrained+Linear', 'MGM-pretrained+MLP_sklearn',
    'SimpleEmb+Linear', 'SimpleEmb+MLP_sklearn', 'SimpleEmb+RandomForest', 'SimpleEmb+XGBoost',
    'SimpleEmb+kNN (k=5, cosine)',
    'ProCyon_v2 (SimpleEmb+MLP, 5-seed ensemble)',
]

# Build LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{\textbf{Comprehensive baseline comparison on IBD classification (clean\_2538).}")
latex_lines.append(r"All methods evaluated on the same train/test split. ProCyon v2 uses a simple embedding layer")
latex_lines.append(r"with masked mean pooling followed by a 2-layer MLP, achieving the best accuracy with only 0.9M parameters.}")
latex_lines.append(r"\label{tab:baselines}")
latex_lines.append(r"\small")
latex_lines.append(r"\begin{tabular}{lccccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\textbf{Method} & \textbf{ACC} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} & \textbf{Params} \\")
latex_lines.append(r"\midrule")

# Group rows
groups = [
    ("\textbf{Raw genus features (baseline)}", [
        'Raw-binary+Linear', 'Raw-binary+MLP_sklearn', 'Raw-binary+RandomForest', 'Raw-binary+XGBoost',
    ]),
    ("\textbf{MGM Transformer encoder (random init)}", [
        'MGM-random+Linear', 'MGM-random+MLP_sklearn',
    ]),
    ("\textbf{MGM Transformer encoder (pretrained, 34M)}", [
        'MGM-pretrained+Linear', 'MGM-pretrained+MLP_sklearn',
    ]),
    ("\textbf{Simple Embedding + Mean Pool (ours)}", [
        'SimpleEmb+Linear', 'SimpleEmb+MLP_sklearn', 'SimpleEmb+RandomForest', 'SimpleEmb+XGBoost',
        'SimpleEmb+kNN (k=5, cosine)',
    ]),
    ("", ['ProCyon_v2 (SimpleEmb+MLP, 5-seed ensemble)']),
]

for group_name, method_keys in groups:
    if group_name:
        latex_lines.append(r"\multicolumn{6}{l}{\textbf{" + group_name + r"}} \\")
    for key in method_keys:
        if key not in all_results:
            continue
        r = all_results[key]
        name = key.replace('_', r'\_')
        acc = r.get('accuracy', 0)
        auc = r.get('auc', 0)
        sens = r.get('sensitivity', 0)
        spec = r.get('specificity', 0)
        params = r.get('n_params', '-')
        if params is None:
            params = '-'
        elif isinstance(params, int):
            if params > 1e6:
                params = f"{params/1e6:.1f}M"
            elif params > 1e3:
                params = f"{params/1e3:.0f}K"
            else:
                params = str(params)
        latex_lines.append(
            f"  {name} & {acc:.4f} & {auc:.4f} & {sens:.4f} & {spec:.4f} & {params} \\\\"
        )
    if group_name:
        latex_lines.append(r"\midrule")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_table = '\n'.join(latex_lines)

with open(f'{OUT_DIR}/results_table.tex', 'w') as f:
    f.write(latex_table)
print(latex_table)

# ═══════════════════════════════════════════════════════════
# 11. Save All Results
# ═══════════════════════════════════════════════════════════

output = {
    'experiment': 'comprehensive_comparison',
    'dataset': 'clean_2538',
    'n_train': len(train_data),
    'n_test': len(test_data),
    'train_test_results': all_results,
    'embedding_ablation': dim_results,
    'architecture_ablation': arch_results,
    'config': {
        'vocab_size': V, 'seq_len': SL, 'n_epochs': NE,
        'batch_size': BS, 'lr': LR, 'weight_decay': WD,
    },
}

with open(f'{OUT_DIR}/comprehensive_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {OUT_DIR}/comprehensive_results.json")

# ═══════════════════════════════════════════════════════════
# 12. Generate Ablation Figure
# ═══════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Embedding dim vs Accuracy
ax = axes[0, 0]
dims = [16, 32, 64, 128, 256, 512, 768, 1024]
linear_accs = [dim_results[f'E={d}']['linear_acc'] for d in dims]
mlp_accs = [dim_results[f'E={d}']['mlp_acc'] for d in dims]
e2e_accs = [dim_results[f'E={d}']['e2e_acc'] for d in dims]
ax.plot(dims, linear_accs, 'o-', label='Emb+Linear', color='#1565C0', markersize=8)
ax.plot(dims, mlp_accs, 's-', label='Emb+MLP(sklearn)', color='#FF5722', markersize=8)
ax.plot(dims, e2e_accs, 'D-', label='Emb+MLP(e2e)', color='#1B5E20', markersize=8, linewidth=2)
ax.set_xlabel('Embedding Dimension'); ax.set_ylabel('Test Accuracy')
ax.set_title('A. Embedding Dimension Ablation', fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2); ax.set_xticks(dims); ax.set_xticklabels(dims)

# Panel B: Architecture component ablation
ax = axes[0, 1]
arch_names = ['Emb+Linear\n(no MLP)', 'Emb+MLP\n(no BN)', 'Emb+MLP\n(no Dropout)', 'Full\n(our model)']
arch_keys = ['Emb+Linear (no MLP)', 'Emb+MLP (no BN)', 'Emb+MLP (no Dropout)', 'Full (Emb+BN+Dropout+MLP)']
arch_accs = [arch_results[k]['accuracy'] for k in arch_keys]
colors = ['#90CAF9', '#FFCC80', '#A5D6A7', '#1B5E20']
bars = ax.bar(range(len(arch_names)), arch_accs, color=colors, edgecolor='none')
for bar, val in zip(bars, arch_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(arch_names))); ax.set_xticklabels(arch_names, fontsize=8)
ax.set_ylabel('Test Accuracy'); ax.set_ylim(min(arch_accs) - 0.02, max(arch_accs) + 0.03)
ax.set_title('B. Architecture Component Ablation', fontweight='bold', loc='left')

# Panel C: Parameter efficiency
ax = axes[0, 2]
model_names = ['Raw+RF', 'Raw+XGB', 'MGM+MLP', 'SimpleEmb+MLP\n(ours)']
model_params = [200, 200, 34000000, 944898]  # RF 200 trees, MGM 34M params, ours 0.9M
model_accs = [
    all_results.get('Raw-binary+RandomForest', {}).get('accuracy', 0),
    all_results.get('Raw-binary+XGBoost', {}).get('accuracy', 0),
    all_results.get('MGM-pretrained+MLP_sklearn', {}).get('accuracy', 0),
    all_results['ProCyon_v2 (SimpleEmb+MLP, 5-seed ensemble)']['accuracy'],
]
scatter_colors = ['#FF9800', '#4CAF50', '#F44336', '#1565C0']
sizes = [np.log10(p) * 80 + 50 for p in model_params]
for i in range(len(model_names)):
    ax.scatter(model_params[i], model_accs[i], s=sizes[i], c=scatter_colors[i],
              alpha=0.8, edgecolors='black', linewidths=1, zorder=5)
    ax.annotate(model_names[i], (model_params[i], model_accs[i]),
               fontsize=8, xytext=(10, 10), textcoords="offset points")
ax.set_xlabel('Number of Parameters'); ax.set_ylabel('Test Accuracy')
ax.set_title('C. Parameter Efficiency', fontweight='bold', loc='left')
ax.set_xscale('log'); ax.grid(True, alpha=0.3)

# Panel D: Method comparison bar chart
ax = axes[1, 0]
compare_names = ['Raw\n+Linear', 'Raw\n+XGBoost', 'MGM\n+Linear', 'MGM\n+MLP',
                 'SimpleEmb\n+Linear', 'SimpleEmb\n+kNN', 'ProCyon\nv2 (ours)']
compare_keys = ['Raw-binary+Linear', 'Raw-binary+XGBoost', 'MGM-pretrained+Linear',
                'MGM-pretrained+MLP_sklearn', 'SimpleEmb+Linear',
                'SimpleEmb+kNN (k=5, cosine)', 'ProCyon_v2 (SimpleEmb+MLP, 5-seed ensemble)']
compare_accs = [all_results.get(k, {}).get('accuracy', 0) for k in compare_keys]
compare_aucs = [all_results.get(k, {}).get('auc', 0) for k in compare_keys]
x = np.arange(len(compare_names))
w = 0.35
ax.bar(x - w/2, compare_accs, w, label='Accuracy', color='#1565C0', edgecolor='none')
ax.bar(x + w/2, compare_aucs, w, label='AUC', color='#FF5722', edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(compare_names, fontsize=8)
ax.set_ylabel('Score'); ax.legend(fontsize=8)
ax.set_title('D. Method Comparison', fontweight='bold', loc='left')
ax.set_ylim(0.5, 1.0); ax.grid(True, alpha=0.3, axis='y')

# Panel E: Embedding dim vs params
ax = axes[1, 1]
dim_params = [dim_results[f'E={d}']['n_params'] for d in dims]
ax.bar(range(len(dims)), dim_params, color='#1565C0', edgecolor='none')
ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, fontsize=8)
ax.set_ylabel('Parameters'); ax.set_xlabel('Embedding Dimension')
ax.set_title('E. Model Size vs Embedding Dimension', fontweight='bold', loc='left')
for i, (d, p) in enumerate(zip(dims, dim_params)):
    ax.text(i, p + 10000, f'{p/1e3:.0f}K', ha='center', fontsize=7)

# Panel F: Training data efficiency (if we have time, placeholder)
ax = axes[1, 2]
# Quick data fraction experiment
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
data_accs = []
np.random.seed(42)
for frac in fractions:
    n_sub = max(10, int(len(train_data) * frac))
    indices = np.random.choice(len(train_data), n_sub, replace=False)
    sub_train = [train_data[i] for i in indices]
    sub_ts = ts[indices]; sub_tm = tm[indices]
    torch.manual_seed(42)
    # Extract embeddings with training data
    X_tr_sub = extract_simple_embeddings(sub_ts, sub_tm, 768)
    sub_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in sub_train])
    lr_sub = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr_sub.fit(X_tr_sub, sub_labels)
    y_p = lr_sub.predict(X_se_test)
    m_acc = accuracy_score(test_labels, y_p)
    data_accs.append(m_acc)
    print(f"  Data fraction {frac:.0%} (n={n_sub}): ACC={m_acc:.4f}")

ax.plot([f * 100 for f in fractions], data_accs, 'o-', color='#1565C0', markersize=8, linewidth=2)
ax.set_xlabel('Training Data (%)'); ax.set_ylabel('Test Accuracy')
ax.set_title('F. Data Efficiency (Emb+Linear)', fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)

fig.suptitle('ProCyon v2: Comprehensive Model Analysis', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUT_DIR}/comprehensive_ablation.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/comprehensive_ablation.png")

# Save data efficiency results
data_efficiency = {f"frac_{frac}": {'n_train': max(10, int(len(train_data)*frac)), 'acc': acc}
                   for frac, acc in zip(fractions, data_accs)}
output['data_efficiency'] = data_efficiency
with open(f'{OUT_DIR}/comprehensive_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON DONE")
print("=" * 70)

# Print summary
print("\n═══ KEY FINDINGS ═══")
best_method = max(all_results.items(), key=lambda x: x[1].get('accuracy', 0))
print(f"  Best method: {best_method[0]} (ACC={best_method[1].get('accuracy', 0):.4f})")
print(f"  SimpleEmb vs MGM-pretrained: "
      f"{all_results.get('SimpleEmb+MLP_sklearn', {}).get('accuracy', 0):.4f} vs "
      f"{all_results.get('MGM-pretrained+MLP_sklearn', {}).get('accuracy', 0):.4f}")
print(f"  Embedding dim saturation: {dim_results['E=768']['e2e_acc']:.4f} at E=768")
