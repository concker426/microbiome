#!/usr/bin/env python3
"""
Structural Baselines + Embedding Biology Analysis
===================================================
Q1: Is Transformer inductive bias wrong for microbiome data?
    → FT-Transformer (feature tokenizer + Transformer, no pretraining)
Q2: Is permutation-invariant set structure the right bias?
    → DeepSets (sum-decomposition, inherently permutation-invariant)
Q3: Does SimpleEmb learn biologically meaningful representations?
    → Genus embedding similarity vs. known functional/phylogenetic groups
"""
import json, os, sys, gc
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

V = 1226; SL = 86; BS = 32; LR_RATE = 1e-3; WD = 1e-4; NE = 50; SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("STRUCTURAL BASELINES + EMBEDDING BIOLOGY")
print("=" * 60)
print(f"Device: {DEVICE}")

# ── Load data ──
train_data = [json.loads(l) for l in open(f'{DATA_DIR}/train_nl.jsonl')]
test_data = [json.loads(l) for l in open(f'{DATA_DIR}/test_nl.jsonl')]
ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')
with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES = json.load(f)['genus_names']

train_labels_arr = np.array([1 if d['label']=='Disease' else 0 for d in train_data])
test_labels_arr = np.array([1 if d['label']=='Disease' else 0 for d in test_data])

# ── DataLoaders ──
class DS(Dataset):
    def __init__(self, data, seqs, masks):
        self.seqs = seqs; self.masks = masks
        self.labels = np.array([1 if d['label']=='Disease' else 0 for d in data])
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

train_ds = DS(train_data, ts, tm); test_ds = DS(test_data, xs, xm)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)

def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import confusion_matrix, precision_score
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'specificity': float(tn/(tn+fp) if (tn+fp)>0 else 0),
    }

def train_eval(model, train_loader, test_loader, epochs=NE):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for gi, gm, y, sw in train_loader:
            gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
            loss = (F.cross_entropy(model(gi, gm), y, reduction='none') * sw).sum() / sw.sum()
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
    preds = np.concatenate(all_preds); probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    n_params = sum(p.numel() for p in model.parameters())
    del model; gc.collect(); torch.cuda.empty_cache()
    m = compute_metrics(labels, preds, probs)
    m['n_params'] = n_params
    return m

# ═══════════════════════════════════════════
# 1. FT-TRANSFORMER (Feature Tokenizer + Transformer)
# ═══════════════════════════════════════════
# Each genus ID → embedding → [CLS] + N tokens → Transformer → CLS → MLP
# This tests: does ANY Transformer work, or is MGM's failure just pretraining?

print("\n[1] FT-Transformer (tabular Transformer, no pretraining)...")

class FTTransformer(nn.Module):
    def __init__(self, vocab=V, dim=128, n_layers=3, n_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))
    def forward(self, ids, mask):
        x = self.emb(ids)  # [B, SL, dim]
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+SL, dim]
        # Create attention mask (True = attend)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=ids.device)
        attn_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=~attn_mask)
        x = self.norm(x[:, 0, :])  # CLS token
        return self.mlp(x)

torch.manual_seed(SEED); np.random.seed(SEED)
ft_model = FTTransformer(vocab=V, dim=128, n_layers=3, n_heads=4, ff_dim=512)
ft_result = train_eval(ft_model, train_loader, test_loader)
print(f"  FT-Transformer: ACC={ft_result['accuracy']:.4f} AUC={ft_result['auc']:.4f} "
      f"F1={ft_result['f1']:.4f} Params={ft_result['n_params']}")

# ═══════════════════════════════════════════
# 2. DEEPSETS (Permutation-Invariant Set Model)
# ═══════════════════════════════════════════
# φ(each genus) → Σ(φ(g)) → ρ(Σ)
# This tests: is the SET structure the right inductive bias?
# SimpleEmb is essentially DeepSets: Embed = φ, Mean = Σ/n, MLP = ρ

print("\n[2] DeepSets (permutation-invariant, no sequential bias)...")

class DeepSets(nn.Module):
    def __init__(self, vocab=V, dim=256, hidden=512):
        super().__init__()
        # φ: element-wise encoder
        self.phi = nn.Sequential(
            nn.Embedding(vocab, dim, padding_idx=0),
        )
        # ρ: set aggregator classifier
        self.rho = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))
    def forward(self, ids, mask):
        x = self.phi(ids)  # [B, SL, dim]
        mf = mask.float().unsqueeze(-1)
        x = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)  # mean pool
        return self.rho(x)

torch.manual_seed(SEED); np.random.seed(SEED)
ds_model = DeepSets(vocab=V, dim=256, hidden=512)
ds_result = train_eval(ds_model, train_loader, test_loader)
print(f"  DeepSets:    ACC={ds_result['accuracy']:.4f} AUC={ds_result['auc']:.4f} "
      f"F1={ds_result['f1']:.4f} Params={ds_result['n_params']}")

# ═══════════════════════════════════════════
# 3. SimpleEmb + MLP (our model, for comparison)
# ═══════════════════════════════════════════
print("\n[3] ProCyon v2 (SimpleEmb + MLP, reference)...")

class SimpleEmb(nn.Module):
    def __init__(self, vocab=V, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))
    def forward(self, ids, mask):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1)
        x = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
        return self.mlp(x)

torch.manual_seed(SEED); np.random.seed(SEED)
se_model = SimpleEmb(vocab=V, dim=256)
se_result = train_eval(se_model, train_loader, test_loader)
print(f"  SimpleEmb:   ACC={se_result['accuracy']:.4f} AUC={se_result['auc']:.4f} "
      f"F1={se_result['f1']:.4f} Params={se_result['n_params']}")

# ═══════════════════════════════════════════
# 4. EMBEDDING BIOLOGY ANALYSIS
# ═══════════════════════════════════════════
print("\n[4] Embedding Biology Analysis...")

# Train SimpleEmb with E=768 to get genus embeddings
torch.manual_seed(42); np.random.seed(42)
bio_model = SimpleEmb(vocab=V, dim=768).to(DEVICE)
opt = torch.optim.AdamW(bio_model.parameters(), lr=LR_RATE, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
for ep in range(NE):
    bio_model.train()
    for gi, gm, y, sw in train_loader:
        gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
        loss = (F.cross_entropy(bio_model(gi, gm), y, reduction='none') * sw).sum() / sw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

# Extract genus embedding matrix
genus_embeddings = bio_model.emb.weight.detach().cpu().numpy()  # [1226, 768]
# Skip padding_idx=0
genus_embeddings = genus_embeddings[1:, :]  # [1225, 768]

# Define known functional groups (genus names from literature)
# SCFA producers, pro-inflammatory, probiotic, etc.
functional_groups = {
    'SCFA_Producers': ['Faecalibacterium', 'Roseburia', 'Eubacterium', 'Coprococcus',
                       'Blautia', 'Ruminococcus', 'Lachnospira', 'Anaerostipes',
                       'Butyricicoccus', 'Subdoligranulum'],
    'Pro_inflammatory': ['Escherichia', 'Proteus', 'Enterobacter', 'Klebsiella',
                         'Fusobacterium', 'Streptococcus', 'Veillonella',
                         'Clostridioides', 'Shigella'],
    'Probiotic': ['Bifidobacterium', 'Lactobacillus', 'Akkermansia'],
    'Mucin_Degraders': ['Akkermansia', 'Bacteroides', 'Ruminococcus'],
    'LPS_Producers': ['Escherichia', 'Klebsiella', 'Enterobacter', 'Proteus',
                      'Pseudomonas', 'Salmonella', 'Shigella'],
}

# Find genus IDs for each group
group_ids = {}
for group_name, genus_list in functional_groups.items():
    ids = []
    for g in genus_list:
        try:
            idx = GENUS_NAMES.index(g) + 1  # +1 because padding_idx=0
            ids.append(idx)
        except ValueError:
            pass
    if len(ids) >= 2:
        group_ids[group_name] = ids
    print(f"  {group_name}: {len(ids)}/{len(genus_list)} genera found in vocab")

# Compute intra-group vs inter-group cosine similarity
def cosine_sim_matrix(embeddings, indices):
    emb = embeddings[indices]  # [N, dim]
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb_norm @ emb_norm.T
    n = len(indices)
    if n < 2: return 0
    # Mean off-diagonal similarity
    mask = ~np.eye(n, dtype=bool)
    return sim[mask].mean()

biology_results = {}
print("\n  Intra-group cosine similarity:")
for group_name, ids in group_ids.items():
    ids_0based = [i-1 for i in ids]  # convert to 0-based index
    intra = cosine_sim_matrix(genus_embeddings, ids_0based)
    biology_results[group_name] = {'n_genera': len(ids), 'intra_sim': float(intra)}
    print(f"    {group_name}: {intra:.4f} (n={len(ids)})")

# Inter-group (random pairs from different groups)
inter_sims = []
rng = np.random.RandomState(42)
all_ids = [i for ids in group_ids.values() for i in ids]
for _ in range(100):
    g1 = rng.choice(all_ids)
    g2 = rng.choice(all_ids)
    if g1 != g2:
        e1 = genus_embeddings[g1-1] / np.linalg.norm(genus_embeddings[g1-1])
        e2 = genus_embeddings[g2-1] / np.linalg.norm(genus_embeddings[g2-1])
        inter_sims.append(np.dot(e1, e2))
mean_inter = float(np.mean(inter_sims))
mean_intra = float(np.mean([v['intra_sim'] for v in biology_results.values() if v['n_genera'] >= 2]))

print(f"\n  Mean intra-group cosine: {mean_intra:.4f}")
print(f"  Mean inter-group cosine: {mean_inter:.4f}")
print(f"  Intra/Inter ratio: {mean_intra/mean_inter:.2f}x")

# ═══════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("STRUCTURAL BASELINES — RESULTS")
print("=" * 70)
print(f"  {'Method':<30s} {'ACC':>8s} {'AUC':>8s} {'F1':>8s} {'Rec':>8s} {'Params':>10s} {'Inductive Bias':>30s}")
print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*30}")

all_structural = {
    'MGM (pretrained Transformer)': {'accuracy': 0.5090, 'auc': 0.4625, 'f1': 0.6058, 'recall': 0.6774, 'n_params': 34_000_000, 'bias': 'Sequential (position-dependent)'},
    'FT-Transformer (no pretrain)': {**ft_result, 'bias': 'Sequential (feature tokenizer)'},
    'DeepSets (set function)': {**ds_result, 'bias': 'Permutation-invariant set'},
    'ProCyon v2 (SimpleEmb+MLP)': {**se_result, 'bias': 'Permutation-invariant set + learned embedding'},
}

for name, r in all_structural.items():
    params = r['n_params']
    ps = f'{params/1e6:.1f}M' if params > 1e6 else (f'{params/1e3:.0f}K' if params > 1e3 else str(params))
    print(f"  {name:<30s} {r['accuracy']:>8.4f} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['recall']:>8.4f} {ps:>10s} {r['bias']:<30s}")

# Save results
results = {
    'experiment': 'structural_baselines',
    'baselines': {k: {kk: vv for kk, vv in v.items() if kk != 'bias'} for k, v in all_structural.items()},
    'embedding_biology': {
        'groups': biology_results,
        'mean_intra_cosine': mean_intra,
        'mean_inter_cosine': mean_inter,
        'intra_inter_ratio': float(mean_intra/mean_inter) if mean_inter > 0 else 0,
    }
}
with open(f'{OUT_DIR}/structural_baselines.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT_DIR}/structural_baselines.json")

# ═══════════════════════════════════════════
# 6. LATEX TABLE
# ═══════════════════════════════════════════
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Structural baselines: inductive bias comparison.}")
tex.append(r"FT-Transformer and DeepSets test whether Transformer architecture")
tex.append(r"or permutation-invariant set structure better suits microbiome data.}")
tex.append(r"\label{tab:structural}")
tex.append(r"\begin{tabular}{lcccccl}")
tex.append(r"\toprule")
tex.append(r"\textbf{Method} & \textbf{ACC} & \textbf{AUC} & \textbf{F1} & \textbf{Rec.} & \textbf{Params} & \textbf{Inductive Bias} \\")
tex.append(r"\midrule")
for name, r in all_structural.items():
    params = r['n_params']
    ps = f'{params/1e6:.1f}M' if params > 1e6 else (f'{params/1e3:.0f}K' if params > 1e3 else str(params))
    tex.append(f"  {name} & {r['accuracy']:.4f} & {r['auc']:.4f} & {r['f1']:.4f} & {r['recall']:.4f} & {ps} & {r['bias']} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

# Biology table
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{\textbf{Embedding biology: intra-group cosine similarity.}")
tex.append(r"Functional groups of genera defined from literature. Higher intra-group")
tex.append(r"similarity indicates the embedding captures known biological relationships.}")
tex.append(r"\label{tab:biology}")
tex.append(r"\begin{tabular}{lcc}")
tex.append(r"\toprule")
tex.append(r"\textbf{Functional Group} & \textbf{N genera} & \textbf{Intra-group Cosine} \\")
tex.append(r"\midrule")
for group_name, info in biology_results.items():
    gn = group_name.replace('_', ' ')
    tex.append(f"  {gn} & {info['n_genera']} & {info['intra_sim']:.4f} \\\\")
tex.append(r"\midrule")
tex.append(f"  Mean intra-group & — & {mean_intra:.4f} \\\\")
tex.append(f"  Mean inter-group & — & {mean_inter:.4f} \\\\")
tex.append(f"  \\textbf{{Intra/Inter ratio}} & — & \\textbf{{{mean_intra/mean_inter:.2f}$\\times$}} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

with open(f'{OUT_DIR}/structural_baselines_table.tex', 'w') as f:
    f.write('\n'.join(tex))
print(f"Saved: {OUT_DIR}/structural_baselines_table.tex")
print("DONE")
