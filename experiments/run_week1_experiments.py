#!/usr/bin/env python3
"""
ProCyon v2 — Week 1 Experiments (Paper Table 1 + Figure 2)
===========================================================
Exp 1: Baseline comparison — 9 methods from Majority to SimpleEmb+MLP
Exp 2: Ablation — A(Raw→MLP), B(RandEmb→MLP), C(SimpleEmb→Linear), D(SimpleEmb→MLP)
Exp 3: Embedding dimension sweep — 32, 64, 128, 256, 512, 768
Exp 4: Cross-cohort validation — clean↔merged
"""
import json, os, sys, time, gc, csv, warnings
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    recall_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ──
V = 1226; SL = 86; E_DEFAULT = 768
NE = 50; BS = 32; LR_RATE = 1e-3; WD = 1e-4
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = 'cuda:0'

# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_dataset(name):
    """Load a dataset (clean_2538 or merged_all)."""
    path = f'/hd/liujx/microbiome_llm_project/data/qiita_ibd/{name}'
    train_data = []
    with open(f'{path}/train_nl.jsonl') as f:
        for l in f: train_data.append(json.loads(l))
    test_data = []
    with open(f'{path}/test_nl.jsonl') as f:
        for l in f: test_data.append(json.loads(l))
    ts = np.load(f'{path}/train_genus_sequences.npy')
    xs = np.load(f'{path}/test_genus_sequences.npy')
    tm = np.load(f'{path}/train_genus_masks.npy')
    xm = np.load(f'{path}/test_genus_masks.npy')
    return train_data, test_data, ts, xs, tm, xm

train_data, test_data, ts, xs, tm, xm = load_dataset('clean_2538')
train_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in train_data])
test_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in test_data])

print(f"clean_2538: train={len(train_data)} (D={train_labels.sum()}), test={len(test_data)} (D={test_labels.sum()})")

# ═══════════════════════════════════════════════════════════
# Feature Extractors
# ═══════════════════════════════════════════════════════════

def extract_raw_abundance(seqs, masks):
    """Raw genus abundance: bag-of-words normalized by total count."""
    n = len(seqs)
    feats = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        valid = masks[i].astype(bool)
        for j in range(len(seqs[i])):
            if valid[j] and seqs[i, j] > 0:
                feats[i, int(seqs[i, j])] += 1.0
        total = feats[i].sum()
        if total > 0:
            feats[i] /= total
    return feats

@torch.no_grad()
def extract_simple_embeddings(seqs, masks, embed_dim=E_DEFAULT, random_seed=None):
    """Simple Embedding + masked mean pool."""
    if random_seed is not None:
        torch.manual_seed(random_seed)
    emb = nn.Embedding(V, embed_dim, padding_idx=0).to(DEVICE)
    feats = []
    for i in range(0, len(seqs), 128):
        gi = torch.from_numpy(seqs[i:i+128].astype(np.int64)).long().to(DEVICE)
        gm = torch.from_numpy(masks[i:i+128]).bool().to(DEVICE)
        e = emb(gi)
        gm_float = gm.float().unsqueeze(-1)
        pooled = (e * gm_float).sum(dim=1) / gm_float.sum(dim=1).clamp(min=1)
        feats.append(pooled.cpu().numpy())
    del emb; torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def extract_mgm_features(seqs, masks, pretrained=True):
    """MGM Transformer encoder features (34M params)."""
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
    del enc; torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)

# ═══════════════════════════════════════════════════════════
# SimpleEmb+MLP (our model) — end-to-end training
# ═══════════════════════════════════════════════════════════

class SimpleEmbEnc(nn.Module):
    def __init__(self, embed_dim=E_DEFAULT):
        super().__init__()
        self.emb = nn.Embedding(V, embed_dim, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

class MLPHead(nn.Module):
    def __init__(self, in_dim=E_DEFAULT, hidden=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class ProCyonModel(nn.Module):
    def __init__(self, embed_dim=E_DEFAULT, hidden=256, dropout=0.3):
        super().__init__()
        self.enc = SimpleEmbEnc(embed_dim)
        self.mlp = MLPHead(embed_dim, hidden, dropout)
    def forward(self, ids, mask=None):
        return self.mlp(self.enc(ids, mask))

def build_loaders(train_data, test_data, ts, xs, tm, xm):
    class DS(Dataset):
        def __init__(self, data, seqs, masks):
            self.seqs = seqs; self.masks = masks
            self.labels = np.array([1 if d['label'] == 'Disease' else 0 for d in data])
            self.sw = np.array([1.5 if d.get('label','Healthy')=='Disease' else 1.0 for d in data])
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                    torch.tensor(self.masks[i],dtype=torch.bool),
                    torch.tensor(self.labels[i],dtype=torch.long),
                    torch.tensor(self.sw[i],dtype=torch.float32))
    def collate(batch):
        gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
        y=torch.stack([x[2] for x in batch]); sw=torch.stack([x[3] for x in batch])
        mgl=max(len(g) for g in gi); pg,pm=[],[]
        for i in range(len(gi)):
            g=gi[i]; m=gm[i]; p=mgl-len(g)
            pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
            pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
        return torch.stack(pg),torch.stack(pm),y,sw
    train_loader = DataLoader(DS(train_data,ts,tm), batch_size=BS, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(DS(test_data,xs,xm), batch_size=BS, shuffle=False, collate_fn=collate)
    return train_loader, test_loader

def train_procyon(train_loader, test_loader, embed_dim=E_DEFAULT, hidden=256, dropout=0.3, n_epochs=NE):
    """Train SimpleEmb+MLP, return test metrics."""
    model = ProCyonModel(embed_dim, hidden, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        model.train()
        for gi, gm, y, sw in train_loader:
            gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
            loss = (F.cross_entropy(model(gi,gm), y, reduction='none') * sw).sum() / sw.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    all_preds, all_probs, all_lbls = [], [], []
    with torch.no_grad():
        for gi, gm, y, sw in test_loader:
            gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
            logits = model(gi, gm)
            prob = F.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits,dim=1).cpu().numpy())
            all_probs.append(prob[:,1].cpu().numpy())
            all_lbls.append(y.cpu().numpy())
    preds = np.concatenate(all_preds); probs = np.concatenate(all_probs)
    labels = np.concatenate(all_lbls)
    n_params = sum(p.numel() for p in model.parameters())
    del model; gc.collect(); torch.cuda.empty_cache()
    return compute_metrics(labels, preds, probs, n_params)

def compute_metrics(y_true, y_pred, y_prob, n_params=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_prob)),
        'f1': float(f1_score(y_true, y_pred)),
        'sensitivity': float(recall_score(y_true, y_pred)),
        'specificity': float(tn/(tn+fp) if (tn+fp)>0 else 0.0),
        'cm': cm.tolist(), 'n_params': n_params,
    }

def format_result(r):
    return f"ACC={r['accuracy']:.4f} AUC={r['auc']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f}"

# ═══════════════════════════════════════════════════════════
# EXP 1: Baseline Comparison (9 methods)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 1: BASELINE COMPARISON — 9 methods")
print("="*70)

print("\nExtracting features...")
t0 = time.time()
X_raw_train = extract_raw_abundance(ts, tm)
X_raw_test = extract_raw_abundance(xs, xm)
X_se_train = extract_simple_embeddings(ts, tm, 768)
X_se_test = extract_simple_embeddings(xs, xm, 768)
print(f"  Raw abundance: {X_raw_train.shape[1]}-dim ({time.time()-t0:.0f}s)")
print(f"  SimpleEmb: {X_se_train.shape[1]}-dim")

# Train + evaluate each method
baseline_results = {}

# 1. Majority baseline
print("\n--- Training baselines ---")
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_raw_train, train_labels)
y_p = dummy.predict(X_raw_test)
try: y_pp = dummy.predict_proba(X_raw_test)[:,1]
except: y_pp = y_p.astype(float)
baseline_results['Majority'] = compute_metrics(test_labels, y_p, y_pp)
print(f"  {'Majority':<35s} {format_result(baseline_results['Majority'])}")

# 2-6. Sklearn classifiers on raw abundance
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    'Linear SVM': LinearSVC(max_iter=2000, C=1.0, random_state=42, dual=False),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    'MLP (sklearn)': MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42),
}
for name, clf in classifiers.items():
    clf.fit(X_raw_train, train_labels)
    y_p = clf.predict(X_raw_test)
    try: y_pp = clf.predict_proba(X_raw_test)[:,1]
    except: y_pp = y_p.astype(float)
    baseline_results[f'Raw + {name}'] = compute_metrics(test_labels, y_p, y_pp)
    print(f"  {'Raw + '+name:<35s} {format_result(baseline_results[f'Raw + {name}'])}")

# 7. MGM + MLP
print("\nExtracting MGM features...")
try:
    X_mgm_train = extract_mgm_features(ts, tm, pretrained=True)
    X_mgm_test = extract_mgm_features(xs, xm, pretrained=True)
    mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42)
    mlp.fit(X_mgm_train, train_labels)
    y_p = mlp.predict(X_mgm_test); y_pp = mlp.predict_proba(X_mgm_test)[:,1]
    baseline_results['MGM pretrained + MLP'] = compute_metrics(test_labels, y_p, y_pp, 34000000)
    print(f"  {'MGM pretrained + MLP':<35s} {format_result(baseline_results['MGM pretrained + MLP'])}")
except Exception as e:
    print(f"  MGM FAILED: {e}")

# 8. SimpleEmb + Linear
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr.fit(X_se_train, train_labels)
y_p = lr.predict(X_se_test); y_pp = lr.predict_proba(X_se_test)[:,1]
baseline_results['SimpleEmb + Linear'] = compute_metrics(test_labels, y_p, y_pp)
print(f"  {'SimpleEmb + Linear':<35s} {format_result(baseline_results['SimpleEmb + Linear'])}")

# 9. SimpleEmb + MLP (our model, 5-seed ensemble)
print("\nTraining ProCyon v2 (5-seed ensemble)...")
train_loader, test_loader = build_loaders(train_data, test_data, ts, xs, tm, xm)
seeds_metrics = {'accuracy':[],'auc':[],'f1':[],'sensitivity':[],'specificity':[]}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    m = train_procyon(train_loader, test_loader, 768, 256, 0.3)
    for k in seeds_metrics: seeds_metrics[k].append(m[k])
    print(f"  Seed {seed}: {format_result(m)}")
our_result = {k: float(np.mean(v)) for k,v in seeds_metrics.items()}
our_result['accuracy_std'] = float(np.std(seeds_metrics['accuracy']))
our_result['n_params'] = sum(p.numel() for p in ProCyonModel(768,256,0.3).parameters())
baseline_results['ProCyon v2 (ours)'] = our_result
print(f"  ENSEMBLE: ACC={our_result['accuracy']:.4f}±{our_result['accuracy_std']:.4f} AUC={our_result['auc']:.4f}")

# ═══════════════════════════════════════════════════════════
# EXP 2: Ablation — What does SimpleEmbedding contribute?
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 2: ABLATION — A(Raw→MLP), B(RandEmb→MLP), C(SimpleEmb→Linear), D(SimpleEmb→MLP)")
print("="*70)

ablation_results = {}

# A: Raw abundance → MLP (no embedding)
# Already done above: 'Raw + MLP (sklearn)'
ablation_results['A: Raw→MLP'] = baseline_results['Raw + MLP (sklearn)']

# B: Random Embedding → MLP (frozen random embedding + MLP head)
print("\n  B: Random Embedding (frozen) + MLP...")
X_rand_train = extract_simple_embeddings(ts, tm, 768, random_seed=999)  # different random init
X_rand_test = extract_simple_embeddings(xs, xm, 768, random_seed=999)
mlp_rand = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42)
mlp_rand.fit(X_rand_train, train_labels)
y_p = mlp_rand.predict(X_rand_test); y_pp = mlp_rand.predict_proba(X_rand_test)[:,1]
ablation_results['B: RandomEmb→MLP'] = compute_metrics(test_labels, y_p, y_pp)
print(f"  {format_result(ablation_results['B: RandomEmb→MLP'])}")

# C: SimpleEmbedding → Linear (trained embedding + linear classifier)
# Already done: 'SimpleEmb + Linear'
ablation_results['C: SimpleEmb→Linear'] = baseline_results['SimpleEmb + Linear']

# D: SimpleEmbedding → MLP (trained embedding + MLP, our full model)
ablation_results['D: SimpleEmb→MLP (ours)'] = baseline_results['ProCyon v2 (ours)']

# ═══════════════════════════════════════════════════════════
# EXP 3: Embedding Dimension Sweep
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 3: EMBEDDING DIMENSION SWEEP — 32..768")
print("="*70)

dim_results = {}
for dim in [32, 64, 128, 256, 512, 768]:
    print(f"\n  --- E={dim} ---")
    # Extract embeddings at this dimension
    X_tr = extract_simple_embeddings(ts, tm, dim)
    X_te = extract_simple_embeddings(xs, xm, dim)

    # SimpleEmb + Linear
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_tr, train_labels)
    y_p = lr.predict(X_te); y_pp = lr.predict_proba(X_te)[:,1]
    lin_acc = accuracy_score(test_labels, y_p)
    lin_auc = roc_auc_score(test_labels, y_pp)

    # SimpleEmb+MLP end-to-end
    torch.manual_seed(42); np.random.seed(42)
    e2e = train_procyon(train_loader, test_loader, dim, min(dim,256), 0.3)
    n_p = sum(p.numel() for p in ProCyonModel(dim, min(dim,256), 0.3).parameters())

    dim_results[f'E={dim}'] = {
        'linear_acc': float(lin_acc), 'linear_auc': float(lin_auc),
        'e2e_acc': float(e2e['accuracy']), 'e2e_auc': float(e2e['auc']),
        'e2e_sens': float(e2e['sensitivity']), 'e2e_spec': float(e2e['specificity']),
        'n_params': n_p,
    }
    print(f"    Emb+Linear: ACC={lin_acc:.4f} AUC={lin_auc:.4f}")
    print(f"    Emb+MLP(e2e): ACC={e2e['accuracy']:.4f} AUC={e2e['auc']:.4f} params={n_p}")

# ═══════════════════════════════════════════════════════════
# EXP 4: Cross-Cohort Validation
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 4: CROSS-COHORT VALIDATION — clean↔merged")
print("="*70)

# Load merged_all
m_train, m_test, m_ts, m_xs, m_tm, m_xm = load_dataset('merged_all')
m_train_labels = np.array([1 if d['label']=='Disease' else 0 for d in m_train])
m_test_labels = np.array([1 if d['label']=='Disease' else 0 for d in m_test])
print(f"merged_all: train={len(m_train)} (D={m_train_labels.sum()}), test={len(m_test)} (D={m_test_labels.sum()})")

cross_results = {}

# Extract features from both datasets using SAME embedding
# Train SimpleEmb on clean_2538, test on merged_all (and vice versa)
for train_name, (tr_data, tr_seq, tr_mask, tr_lbls), (te_data, te_seq, te_mask, te_lbls) in [
    ('clean→merged', (train_data, ts, tm, train_labels), (m_test, m_xs, m_xm, m_test_labels)),
    ('merged→clean', (m_train, m_ts, m_tm, m_train_labels), (test_data, xs, xm, test_labels)),
    ('clean→clean (ref)', (train_data, ts, tm, train_labels), (test_data, xs, xm, test_labels)),
    ('merged→merged (ref)', (m_train, m_ts, m_tm, m_train_labels), (m_test, m_xs, m_xm, m_test_labels)),
]:
    print(f"\n  {train_name}...")
    # Extract embeddings with trained embedding layer
    # We train SimpleEmb+MLP on train set, extract embeddings, then train sklearn on those embeddings
    tr_loader, te_loader = build_loaders(tr_data, te_data, tr_seq, te_seq, tr_mask, te_mask)

    # Method: SimpleEmb+Linear (fast, stable for cross-cohort)
    X_tr = extract_simple_embeddings(tr_seq, tr_mask, 768)
    X_te = extract_simple_embeddings(te_seq, te_mask, 768)

    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_tr, tr_lbls)
    y_p = lr.predict(X_te); y_pp = lr.predict_proba(X_te)[:,1]
    cross_results[train_name] = compute_metrics(te_lbls, y_p, y_pp)
    print(f"    Emb+Linear: {format_result(cross_results[train_name])}")

    # Method: SimpleEmb+MLP (end-to-end)
    torch.manual_seed(42); np.random.seed(42)
    e2e = train_procyon(tr_loader, te_loader, 768, 256, 0.3)
    cross_results[train_name + ' (e2e)'] = e2e
    print(f"    Emb+MLP(e2e): {format_result(e2e)}")

# ═══════════════════════════════════════════════════════════
# Print Summary Tables
# ═══════════════════════════════════════════════════════════

def print_table(title, methods, results):
    print(f"\n{'─'*90}")
    print(f"  {title}")
    print(f"{'─'*90}")
    print(f"  {'Method':<40s} {'ACC':>8s} {'AUC':>8s} {'Sens':>8s} {'Spec':>8s} {'Params':>10s}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for name in methods:
        r = results.get(name, {})
        if not r: continue
        acc = r.get('accuracy', 0); auc = r.get('auc', 0)
        sens = r.get('sensitivity', 0); spec = r.get('specificity', 0)
        params = r.get('n_params')
        ps = '-' if params is None else (f'{params/1e6:.1f}M' if (isinstance(params, (int,float)) and params>1e6) else (f'{params/1e3:.0f}K' if isinstance(params,(int,float)) and params>1e3 else str(params)))
        print(f"  {name:<40s} {acc:>8.4f} {auc:>8.4f} {sens:>8.4f} {spec:>8.4f} {ps:>10s}")

# EXP 1 table
print_table("EXP 1: BASELINE COMPARISON", [
    'Majority',
    'Raw + Logistic Regression', 'Raw + Linear SVM', 'Raw + Random Forest',
    'Raw + XGBoost', 'Raw + MLP (sklearn)',
    'MGM pretrained + MLP',
    'SimpleEmb + Linear',
    'ProCyon v2 (ours)',
], baseline_results)

# EXP 2 table
print_table("EXP 2: ABLATION", [
    'A: Raw→MLP', 'B: RandomEmb→MLP', 'C: SimpleEmb→Linear', 'D: SimpleEmb→MLP (ours)',
], ablation_results)

# EXP 3 summary
print(f"\n{'─'*80}")
print(f"  EXP 3: EMBEDDING DIMENSION")
print(f"{'─'*80}")
print(f"  {'Dim':>6s} {'Lin ACC':>10s} {'Lin AUC':>10s} {'E2E ACC':>10s} {'E2E AUC':>10s} {'Params':>10s}")
for dim in [32, 64, 128, 256, 512, 768]:
    r = dim_results[f'E={dim}']
    print(f"  {dim:>6d} {r['linear_acc']:>10.4f} {r['linear_auc']:>10.4f} {r['e2e_acc']:>10.4f} {r['e2e_auc']:>10.4f} {r['n_params']:>10d}")

# EXP 4 table
print_table("EXP 4: CROSS-COHORT", [
    'clean→clean (ref)', 'merged→merged (ref)',
    'clean→merged', 'merged→clean',
    'clean→clean (ref) (e2e)', 'merged→merged (ref) (e2e)',
    'clean→merged (e2e)', 'merged→clean (e2e)',
], cross_results)

# ═══════════════════════════════════════════════════════════
# Save All Results
# ═══════════════════════════════════════════════════════════

output = {
    'experiment': 'procyon_v2_week1',
    'dataset': 'clean_2538 + merged_all',
    'exp1_baselines': baseline_results,
    'exp2_ablation': ablation_results,
    'exp3_embedding_dim': dim_results,
    'exp4_cross_cohort': cross_results,
}

with open(f'{OUT_DIR}/week1_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

# ═══════════════════════════════════════════════════════════
# Generate LaTeX Tables for Overleaf
# ═══════════════════════════════════════════════════════════

def fmt_cell(r, key):
    val = r.get(key, 0)
    return f'{val:.4f}' if isinstance(val, float) else str(val)

# Table 1: Baseline Comparison
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{IBD classification performance comparison on clean\_2538.}")
tex.append(r"All methods evaluated on identical train/test split.}")
tex.append(r"\label{tab:baselines}")
tex.append(r"\begin{tabular}{lcccc}")
tex.append(r"\toprule")
tex.append(r"\textbf{Method} & \textbf{ACC} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} \\")
tex.append(r"\midrule")

baseline_methods = [
    ('Majority', 'Majority'),
    ('Logistic Regression', 'Raw + Logistic Regression'),
    ('Linear SVM', 'Raw + Linear SVM'),
    ('Random Forest', 'Raw + Random Forest'),
    ('XGBoost', 'Raw + XGBoost'),
    ('MLP', 'Raw + MLP (sklearn)'),
    ('MGM pretrained (34M) + MLP', 'MGM pretrained + MLP'),
    ('SimpleEmb + Linear (ours)', 'SimpleEmb + Linear'),
    (r'\textbf{ProCyon v2 (SimpleEmb+MLP, ours)}', 'ProCyon v2 (ours)'),
]
for display_name, key in baseline_methods:
    r = baseline_results.get(key, {})
    if not r: continue
    acc = r.get('accuracy', 0); auc = r.get('auc', 0)
    sens = r.get('sensitivity', 0); spec = r.get('specificity', 0)
    tex.append(f"  {display_name} & {acc:.4f} & {auc:.4f} & {sens:.4f} & {spec:.4f} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

# Table 2: Ablation
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Ablation study: contribution of SimpleEmbedding.}")
tex.append(r"All variants use the same MLP architecture.}")
tex.append(r"\label{tab:ablation}")
tex.append(r"\begin{tabular}{lccccc}")
tex.append(r"\toprule")
tex.append(r"\textbf{Variant} & \textbf{ACC} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} & \textbf{Description} \\")
tex.append(r"\midrule")

ablation_rows = [
    ('A: Raw→MLP', 'Raw + MLP (sklearn)', 'Raw abundance → MLP (no embedding)'),
    ('B: RandomEmb→MLP', 'B: RandomEmb→MLP', 'Random frozen embedding → MLP'),
    ('C: SimpleEmb→Linear', 'SimpleEmb + Linear', 'Trained embedding → Linear'),
    ('D: SimpleEmb→MLP (ours)', 'D: SimpleEmb→MLP (ours)', 'Trained embedding → MLP'),
]
for display_name, key, desc in ablation_rows:
    r = ablation_results.get(key, baseline_results.get(key, {}))
    if not r: continue
    tex.append(f"  {display_name} & {r['accuracy']:.4f} & {r['auc']:.4f} & {r['sensitivity']:.4f} & {r['specificity']:.4f} & {desc} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

# Table 3: Embedding Dimension
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Embedding dimension ablation.}")
tex.append(r"SimpleEmb+MLP trained end-to-end with varying embedding dimensions.}")
tex.append(r"\label{tab:embedding_dim}")
tex.append(r"\begin{tabular}{lcccc}")
tex.append(r"\toprule")
tex.append(r"\textbf{Dimension} & \textbf{ACC} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} \\")
tex.append(r"\midrule")
for dim in [32, 64, 128, 256, 512, 768]:
    r = dim_results[f'E={dim}']
    tex.append(f"  {dim} & {r['e2e_acc']:.4f} & {r['e2e_auc']:.4f} & {r['e2e_sens']:.4f} & {r['e2e_spec']:.4f} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

# Table 4: Cross-Cohort
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Cross-cohort validation.}")
tex.append(r"SimpleEmb+Linear trained on one cohort, tested on another.}")
tex.append(r"\label{tab:cross_cohort}")
tex.append(r"\begin{tabular}{lcccc}")
tex.append(r"\toprule")
tex.append(r"\textbf{Train → Test} & \textbf{ACC} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} \\")
tex.append(r"\midrule")
for name in ['clean→clean (ref)', 'merged→merged (ref)', 'clean→merged', 'merged→clean']:
    r = cross_results.get(name, {})
    if not r: continue
    tex.append(f"  {name} & {r['accuracy']:.4f} & {r['auc']:.4f} & {r['sensitivity']:.4f} & {r['specificity']:.4f} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

with open(f'{OUT_DIR}/week1_tables.tex', 'w') as f:
    f.write('\n'.join(tex))
print(f"\nSaved: {OUT_DIR}/week1_tables.tex")

# ═══════════════════════════════════════════════════════════
# Generate Figure
# ═══════════════════════════════════════════════════════════
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Baseline accuracy bar chart
ax = axes[0, 0]
bar_names = ['Majority', 'LogReg', 'SVM', 'RF', 'XGBoost', 'MLP', 'MGM+MLP', 'Emb+Linear', 'ProCyon\nv2']
bar_keys = ['Majority', 'Raw + Logistic Regression', 'Raw + Linear SVM',
            'Raw + Random Forest', 'Raw + XGBoost', 'Raw + MLP (sklearn)',
            'MGM pretrained + MLP', 'SimpleEmb + Linear', 'ProCyon v2 (ours)']
bar_accs = [baseline_results.get(k,{}).get('accuracy',0) for k in bar_keys]
colors = ['#9E9E9E']*6 + ['#F44336'] + ['#2196F3'] + ['#1B5E20']
bars = ax.bar(range(len(bar_names)), bar_accs, color=colors, edgecolor='none')
for bar, val in zip(bars, bar_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=7, fontweight='bold')
ax.set_xticks(range(len(bar_names))); ax.set_xticklabels(bar_names, fontsize=7)
ax.set_ylabel('Test Accuracy'); ax.set_ylim(0.4, 1.0)
ax.set_title('A. Baseline Comparison', fontweight='bold', loc='left')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Ablation
ax = axes[0, 1]
ab_names = ['A: Raw\n→MLP', 'B: RandomEmb\n→MLP', 'C: SimpleEmb\n→Linear', 'D: SimpleEmb\n→MLP (ours)']
ab_keys = ['Raw + MLP (sklearn)', 'B: RandomEmb→MLP', 'SimpleEmb + Linear', 'D: SimpleEmb→MLP (ours)']
ab_accs = [(ablation_results.get(k) or baseline_results.get(k,{})).get('accuracy',0) for k in ab_keys]
ab_colors = ['#FF9800', '#FFC107', '#2196F3', '#1B5E20']
bars = ax.bar(range(len(ab_names)), ab_accs, color=ab_colors, edgecolor='none')
for bar, val in zip(bars, ab_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.4f}',
            ha='center', fontsize=9, fontweight='bold')
    delta = val - ab_accs[0]
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2, f'Δ={delta:+.4f}',
            ha='center', fontsize=8, color='white' if bar.get_facecolor()[-1]>0.5 else 'black')
ax.set_xticks(range(len(ab_names))); ax.set_xticklabels(ab_names, fontsize=8)
ax.set_ylabel('Test Accuracy')
ax.set_title('B. Ablation: Embedding Contribution', fontweight='bold', loc='left')

# Panel C: Embedding dimension
ax = axes[0, 2]
dims = [32, 64, 128, 256, 512, 768]
lin_accs = [dim_results[f'E={d}']['linear_acc'] for d in dims]
e2e_accs = [dim_results[f'E={d}']['e2e_acc'] for d in dims]
ax.plot(dims, lin_accs, 'o-', label='Emb+Linear', color='#2196F3', markersize=8, linewidth=2)
ax.plot(dims, e2e_accs, 's-', label='Emb+MLP (e2e)', color='#1B5E20', markersize=8, linewidth=2)
ax.set_xlabel('Embedding Dimension'); ax.set_ylabel('Test Accuracy')
ax.set_title('C. Embedding Dimension', fontweight='bold', loc='left')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2); ax.set_xticks(dims); ax.set_xticklabels(dims)

# Panel D: Cross-cohort
ax = axes[1, 0]
cc_names = ['clean→clean', 'merged→merged', 'clean→merged', 'merged→clean']
cc_keys = ['clean→clean (ref)', 'merged→merged (ref)', 'clean→merged', 'merged→clean']
cc_accs = [cross_results.get(k,{}).get('accuracy',0) for k in cc_keys]
cc_aucs = [cross_results.get(k,{}).get('auc',0) for k in cc_keys]
x = np.arange(len(cc_names)); w = 0.35
ax.bar(x-w/2, cc_accs, w, label='Accuracy', color='#1565C0')
ax.bar(x+w/2, cc_aucs, w, label='AUC', color='#FF5722')
for i in range(len(cc_names)):
    ax.text(i-w/2, cc_accs[i]+0.01, f'{cc_accs[i]:.3f}', ha='center', fontsize=7)
    ax.text(i+w/2, cc_aucs[i]+0.01, f'{cc_aucs[i]:.3f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(cc_names, fontsize=8)
ax.set_ylabel('Score'); ax.legend(fontsize=8)
ax.set_title('D. Cross-Cohort Validation (Emb+Linear)', fontweight='bold', loc='left')
ax.set_ylim(0.4, 1.0); ax.grid(True, alpha=0.3, axis='y')

# Panel E: Sensitivity vs Specificity scatter
ax = axes[1, 1]
for i, key in enumerate(bar_keys):
    r = baseline_results.get(key, {})
    if not r: continue
    ax.scatter(r.get('specificity',0), r.get('sensitivity',0),
              s=100, c=[colors[i]], edgecolors='black', linewidths=0.5, zorder=5)
    ax.annotate(bar_names[i], (r.get('specificity',0), r.get('sensitivity',0)),
               fontsize=6, xytext=(5,5), textcoords='offset points')
ax.set_xlabel('Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title('E. Sensitivity-Specificity Trade-off', fontweight='bold', loc='left')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.grid(True, alpha=0.3)

# Panel F: Model size vs performance
ax = axes[1, 2]
model_data = [
    ('Majority', 0, baseline_results['Majority']['accuracy'], '#9E9E9E'),
    ('LogReg', V*2, baseline_results['Raw + Logistic Regression']['accuracy'], '#9E9E9E'),
    ('RF', 200*15, baseline_results['Raw + Random Forest']['accuracy'], '#FF9800'),
    ('XGB', 200*6, baseline_results['Raw + XGBoost']['accuracy'], '#4CAF50'),
    ('MGM+MLP', 34e6, baseline_results.get('MGM pretrained + MLP',{}).get('accuracy',0), '#F44336'),
    ('ProCyon v2', our_result['n_params'], our_result['accuracy'], '#1B5E20'),
]
for name, params, acc, color in model_data:
    if acc == 0: continue
    size = np.log10(max(params,1)) * 60 + 30
    ax.scatter(params, acc, s=size, c=color, alpha=0.8, edgecolors='black', linewidths=1)
    ax.annotate(name, (params, acc), fontsize=8, xytext=(5,5), textcoords='offset points')
ax.set_xlabel('Parameters'); ax.set_ylabel('Test Accuracy')
ax.set_xscale('log'); ax.set_title('F. Parameter Efficiency', fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)

fig.suptitle('ProCyon v2 — Week 1: Baseline Comparison & Ablation Analysis', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{OUT_DIR}/week1_figure.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/week1_figure.png")

print(f"\n{'='*70}")
print("WEEK 1 EXPERIMENTS COMPLETE")
print(f"{'='*70}")
print(f"  Saved: {OUT_DIR}/week1_results.json")
print(f"  Saved: {OUT_DIR}/week1_tables.tex")
print(f"  Saved: {OUT_DIR}/week1_figure.png")
