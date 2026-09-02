#!/usr/bin/env python3
"""Extended SimpleEmb-MAP ablation, part 2 (runs unattended via nohup).

PART 1 - holdout ablations (identical protocol to ablation_simpleemb_map.py:
  E=768, MLP 256, 50 epochs, BS=32, AdamW 1e-3/wd 1e-4, cosine, 5 seeds, w=1.5 Disease):
    A3: gated (attention) pooling instead of mean pooling
    A6: token dropout (p=0.15) during training (DeepMicro-style denoising)
    A8: linear head (end-to-end trained embedding + Linear) to isolate the MLP

PART 2 - retention test for A1 rank-bin under the paper's grouped-CV protocol
  (decontaminated_groupcv: all 826 samples, sequence-hash groups,
   3 seeds x StratifiedGroupKFold(5) = 15 folds, E=512, BS=64):
  A0 vs A1 per fold, paired-bootstrap 95% CIs, count non-overlapping folds.
"""
import hashlib, json, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = "/hd/liujx/microbiome_llm_project/data/qiita_ibd"
C = f"{BASE}/clean_2538"
AN = "/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V = 1226

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- data (shared) ----------
vocab = json.load(open("/hd/liujx/microbiome_data/qiita_pretrain/genus_vocab.json"))
n2t = {k[3:]: v for k, v in vocab.items() if k.startswith("g__")}
qi = json.load(open(f"{BASE}/qiita_ibd_info.json")); gnames = qi["genus_names"]
sv = np.load(f"{BASE}/study_2538_vectors.npy")
sv_pos = {s: i for i, s in enumerate(json.load(open(f"{BASE}/study_2538_ids.json")))}

def load_split(name):
    data = [json.loads(l) for l in open(f"{C}/{name}_nl.jsonl")]
    seqs = np.load(f"{C}/{name}_genus_sequences.npy")
    masks = np.load(f"{C}/{name}_genus_masks.npy")
    y = np.array([1 if d["label"] == "Disease" else 0 for d in data])
    abund = np.zeros_like(seqs, dtype=np.float32)
    for j, d in enumerate(data):
        row = sv[sv_pos[d["sample_id"]]]
        tok2val = {}
        for g in np.where(row > 0)[0]:
            t = n2t.get(gnames[g])
            if t is not None and t < V:
                tok2val[t] = row[g]
        for k, t in enumerate(seqs[j]):
            if masks[j, k]:
                abund[j, k] = tok2val.get(int(t), 0.0)
    return data, seqs, masks, y, abund

def rank_bins(seqs, masks):
    rb = np.zeros_like(seqs, dtype=np.int64)
    for j in range(len(seqs)):
        pos = np.where(masks[j])[0]; N = len(pos)
        if N: rb[j, pos] = np.minimum((np.arange(N) / N * 32).astype(np.int64), 31)
    return rb

tr_d, tr_s, tr_m, tr_y, tr_a = load_split("train")
te_d, te_s, te_m, te_y, te_a = load_split("test")
tr_rb, te_rb = rank_bins(tr_s, tr_m), rank_bins(te_s, te_m)
all_d = tr_d + te_d
all_s = np.concatenate([tr_s, te_s]); all_m = np.concatenate([tr_m, te_m])
all_y = np.concatenate([tr_y, te_y]); all_rb = np.concatenate([tr_rb, te_rb])
print(f"data: train {len(tr_y)} test {len(te_y)} all {len(all_y)}", flush=True)

# ═══════════ PART 1: holdout A3/A6/A8 (E=768 protocol) ═══════════
E1, HID, NE1, BS1, LR, WD = 768, 256, 50, 32, 1e-3, 1e-4
SEEDS1 = [42, 123, 456, 789, 1024]

class GatePool(nn.Module):
    def __init__(self, e):
        super().__init__(); self.lin = nn.Linear(e, 1)
    def forward(self, x, mask):
        sc = torch.tanh(self.lin(x)).squeeze(-1)
        sc = sc.masked_fill(~mask.bool(), float("-inf"))
        w = F.softmax(sc, dim=1)
        return (x * w.unsqueeze(-1)).sum(1)

class MeanPool(nn.Module):
    def forward(self, x, mask):
        mf = mask.float().unsqueeze(-1)
        return (x * mf).sum(1) / mf.sum(1).clamp(min=1)

class MLPHead(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.fc1 = nn.Linear(e, HID); self.bn1 = nn.BatchNorm1d(HID)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(HID, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class Model1(nn.Module):
    def __init__(self, variant):  # variant in {"A0","A3","A8"}
        super().__init__()
        self.emb = nn.Embedding(V, E1, padding_idx=0)
        self.pool = GatePool(E1) if variant == "A3" else MeanPool()
        self.head = nn.Linear(E1, 2) if variant == "A8" else MLPHead(E1)
    def forward(self, ids, mask):
        return self.head(self.pool(self.emb(ids), mask))

def t64(x): return torch.tensor(x).long().to(DEVICE)
Xtr, Mtr = t64(tr_s), t64(tr_m); Xte, Mte = t64(te_s), t64(te_m)
Ytr = torch.tensor(tr_y).long().to(DEVICE)
Wtr = torch.tensor(np.where(tr_y == 1, 1.5, 1.0)).float().to(DEVICE)

def run_holdout(variant, tokdrop=0.0):
    mets = []
    for sd in SEEDS1:
        set_seed(sd)
        m = Model1(variant).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE1)
        for ep in range(NE1):
            m.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), BS1):
                idx = perm[i:i + BS1]
                if len(idx) < 2: continue
                mk = Mtr[idx]
                if tokdrop > 0:  # A6: drop present tokens during training
                    drop = (torch.rand(mk.shape, device=DEVICE) < tokdrop) & mk.bool()
                    mk = mk.bool() & ~drop
                loss = (F.cross_entropy(m(Xtr[idx], mk), Ytr[idx], reduction="none") * Wtr[idx]).sum() / Wtr[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(Xte, Mte), 1)[:, 1].cpu().numpy()
        mets.append([accuracy_score(te_y, (p > .5).astype(int)),
                     roc_auc_score(te_y, p), average_precision_score(te_y, p)])
    a = np.array(mets)
    return {"acc": a[:, 0].mean(), "acc_std": a[:, 0].std(),
            "auc": a[:, 1].mean(), "auc_std": a[:, 1].std(),
            "auprc": a[:, 2].mean(), "auprc_std": a[:, 2].std()}

res1 = {}
for name, kw in [("A3_gated_pooling", dict(variant="A3")),
                 ("A6_token_dropout", dict(variant="A0", tokdrop=0.15)),
                 ("A8_linear_head", dict(variant="A8"))]:
    t0 = time.time()
    res1[name] = run_holdout(**kw)
    r = res1[name]
    print(f"{name}: ACC {r['acc']:.4f}±{r['acc_std']:.4f} | AUROC {r['auc']:.4f}±{r['auc_std']:.4f} | AUPRC {r['auprc']:.4f}±{r['auprc_std']:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(res1, open(f"{AN}/ablation_a3_a6_a8.json", "w"), indent=2)
print("PART1 saved", flush=True)

# ═══════════ PART 2: 15-fold grouped CV, A0 vs A1 (E=512 protocol) ═══════════
E2, NE2, BS2 = 512, 50, 64
SEEDS2 = [42, 123, 456]

class Model2(nn.Module):
    def __init__(self, use_rank):
        super().__init__()
        self.emb = nn.Embedding(V, E2, padding_idx=0)
        self.use_rank = use_rank
        if use_rank: self.rank_emb = nn.Embedding(32, E2)
        self.fc1 = nn.Linear(E2, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, ids, mask, rb=None):
        x = self.emb(ids)
        if self.use_rank: x = x + self.rank_emb(rb)
        mf = mask.float().unsqueeze(-1)
        pooled = (x * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.fc2(F.dropout(F.relu(self.bn1(self.fc1(pooled))), 0.3, self.training))

def train_fold(use_rank, ti, vi, seed):
    set_seed(seed)
    m = Model2(use_rank).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE2)
    Xti, Mti = t64(all_s[ti]), t64(all_m[ti])
    Rti = t64(all_rb[ti]) if use_rank else None
    y = torch.tensor(all_y[ti]).long().to(DEVICE)
    w = torch.tensor(np.where(all_y[ti] == 1, 1.5, 1.0)).float().to(DEVICE)
    for ep in range(NE2):
        m.train()
        perm = torch.randperm(len(ti))
        for i in range(0, len(ti), BS2):
            idx = perm[i:i + BS2]
            if len(idx) < 2: continue
            logits = m(Xti[idx], Mti[idx], Rti[idx] if Rti is not None else None)
            loss = (F.cross_entropy(logits, y[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(t64(all_s[vi]), t64(all_m[vi]),
                            t64(all_rb[vi]) if use_rank else None), 1)[:, 1].cpu().numpy()
    return p

def seq_hash(seq, mask):
    packed = np.ascontiguousarray(np.column_stack((seq, mask)))
    return hashlib.sha256(packed.tobytes()).hexdigest()

groups = np.array([seq_hash(s, mk) for s, mk in zip(all_s, all_m)])
folds = []
for seed in SEEDS2:
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(sp.split(all_s, all_y, groups), start=1):
        assert not (set(groups[ti]) & set(groups[vi]))
        folds.append((seed, fold, ti, vi))
print(f"PART2: {len(folds)} folds", flush=True)

per_fold, p0_all, p1_all, y_all, fold_id = [], [], [], [], []
for seed, fold, ti, vi in folds:
    t0 = time.time()
    fs = seed * 100 + fold
    p0 = train_fold(False, ti, vi, fs)
    p1 = train_fold(True, ti, vi, fs)
    yv = all_y[vi]
    r = {
        "seed": seed, "fold": fold,
        "auc0": roc_auc_score(yv, p0), "auc1": roc_auc_score(yv, p1),
        "auprc0": average_precision_score(yv, p0), "auprc1": average_precision_score(yv, p1),
        "acc0": accuracy_score(yv, (p0 > .5).astype(int)), "acc1": accuracy_score(yv, (p1 > .5).astype(int)),
    }
    # paired bootstrap 95% CIs (1000 resamples), non-overlap = lo1 > hi0
    rng = np.random.RandomState(fs); n = len(vi)
    wins_auc = wins_auprc = 0
    for _ in range(1000):
        b = rng.randint(0, n, n)
        if len(np.unique(yv[b])) < 2: continue
        a0 = roc_auc_score(yv[b], p0[b]); a1 = roc_auc_score(yv[b], p1[b])
        u0 = average_precision_score(yv[b], p0[b]); u1 = average_precision_score(yv[b], p1[b])
        if a1 > a0: wins_auc += 1
        if u1 > u0: wins_auprc += 1
    r["boot_win_rate_auc"] = wins_auc / 1000.0
    r["boot_win_rate_auprc"] = wins_auprc / 1000.0
    r["auc1_gt_auc0"] = bool(r["auc1"] > r["auc0"])
    r["auprc1_gt_auprc0"] = bool(r["auprc1"] > r["auprc0"])
    per_fold.append(r)
    p0_all.extend(p0.tolist()); p1_all.extend(p1.tolist()); y_all.extend(yv.tolist()); fold_id.extend([len(fold_id)] * n)
    print(f"  seed{seed} f{fold}: AUC {r['auc0']:.4f}->{r['auc1']:.4f} AUPRC {r['auprc0']:.4f}->{r['auprc1']:.4f} winAUC {r['boot_win_rate_auc']:.2f} ({time.time()-t0:.0f}s)", flush=True)

pf = {k: float(np.mean([f[k] for f in per_fold])) for k in
      ["auc0", "auc1", "auprc0", "auprc1", "acc0", "acc1", "boot_win_rate_auc", "boot_win_rate_auprc"]}
pf_std = {k + "_std": float(np.std([f[k] for f in per_fold])) for k in ["auc0", "auc1", "auprc0", "auprc1"]}
summary = {
    "per_fold": per_fold,
    "mean": {**pf, **pf_std},
    "n_folds_auc1_wins": int(sum(f["auc1_gt_auc0"] for f in per_fold)),
    "n_folds_auprc1_wins": int(sum(f["auprc1_gt_auprc0"] for f in per_fold)),
    "n_folds_boot_auc_ge_95": int(sum(f["boot_win_rate_auc"] >= 0.95 for f in per_fold)),
    "n_folds_boot_auprc_ge_95": int(sum(f["boot_win_rate_auprc"] >= 0.95 for f in per_fold)),
    "retention_rule": "A1 retained iff boot win-rate >= 0.95 in >= 10/15 folds for AUC, AUPRC not degraded",
    "auprc_not_degraded": bool(pf["auprc1"] >= pf["auprc0"] - 0.002),
}
json.dump(summary, open(f"{AN}/groupcv_a0_vs_a1.json", "w"), indent=2)
print("PART2 summary:", json.dumps({k: v for k, v in summary.items() if k != "per_fold"}, indent=1), flush=True)

open(f"{AN}/ablation_ext.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " completed\n")
print("ALL DONE", flush=True)
