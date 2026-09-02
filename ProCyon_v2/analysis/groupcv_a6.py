#!/usr/bin/env python3
"""15-fold grouped-CV retention test for A6 token dropout (p=0.15) vs A0.
Identical protocol to groupcv_a0_vs_a1 in ablation_ext.py."""
import hashlib, json, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = "/hd/liujx/microbiome_llm_project/data/qiita_ibd"
C = f"{BASE}/clean_2538"
AN = "/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E2, NE2, BS2, LR, WD = 1226, 512, 50, 64, 1e-3, 1e-4
SEEDS2 = [42, 123, 456]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

tr_d = [json.loads(l) for l in open(f"{C}/train_nl.jsonl")]
te_d = [json.loads(l) for l in open(f"{C}/test_nl.jsonl")]
all_s = np.concatenate([np.load(f"{C}/train_genus_sequences.npy"), np.load(f"{C}/test_genus_sequences.npy")])
all_m = np.concatenate([np.load(f"{C}/train_genus_masks.npy"), np.load(f"{C}/test_genus_masks.npy")])
all_y = np.concatenate([np.array([1 if d["label"] == "Disease" else 0 for d in tr_d]),
                        np.array([1 if d["label"] == "Disease" else 0 for d in te_d])])

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E2, padding_idx=0)
        self.fc1 = nn.Linear(E2, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, ids, mask):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1)
        pooled = (x * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.fc2(F.dropout(F.relu(self.bn1(self.fc1(pooled))), 0.3, self.training))

def t64(x): return torch.tensor(x).long().to(DEVICE)

def train_fold(tokdrop, ti, vi, seed):
    set_seed(seed)
    m = Model2().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE2)
    Xti, Mti = t64(all_s[ti]), t64(all_m[ti])
    y = torch.tensor(all_y[ti]).long().to(DEVICE)
    w = torch.tensor(np.where(all_y[ti] == 1, 1.5, 1.0)).float().to(DEVICE)
    for ep in range(NE2):
        m.train()
        perm = torch.randperm(len(ti))
        for i in range(0, len(ti), BS2):
            idx = perm[i:i + BS2]
            if len(idx) < 2: continue
            mk = Mti[idx]
            if tokdrop > 0:
                drop = (torch.rand(mk.shape, device=DEVICE) < tokdrop) & mk.bool()
                mk = mk.bool() & ~drop
            loss = (F.cross_entropy(m(Xti[idx], mk), y[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(t64(all_s[vi]), t64(all_m[vi])), 1)[:, 1].cpu().numpy()
    return p

def seq_hash(seq, mask):
    return hashlib.sha256(np.ascontiguousarray(np.column_stack((seq, mask))).tobytes()).hexdigest()

groups = np.array([seq_hash(s, mk) for s, mk in zip(all_s, all_m)])
per_fold = []
t00 = time.time()
for seed in SEEDS2:
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(sp.split(all_s, all_y, groups), start=1):
        assert not (set(groups[ti]) & set(groups[vi]))
        fs = seed * 100 + fold
        p0 = train_fold(0.0, ti, vi, fs)
        p6 = train_fold(0.15, ti, vi, fs)
        yv = all_y[vi]
        r = {"seed": seed, "fold": fold,
             "auc0": roc_auc_score(yv, p0), "auc6": roc_auc_score(yv, p6),
             "auprc0": average_precision_score(yv, p0), "auprc6": average_precision_score(yv, p6),
             "acc0": accuracy_score(yv, (p0 > .5).astype(int)), "acc6": accuracy_score(yv, (p6 > .5).astype(int))}
        rng = np.random.RandomState(fs); n = len(vi)
        wa = wu = 0
        for _ in range(1000):
            b = rng.randint(0, n, n)
            if len(np.unique(yv[b])) < 2: continue
            if roc_auc_score(yv[b], p6[b]) > roc_auc_score(yv[b], p0[b]): wa += 1
            if average_precision_score(yv[b], p6[b]) > average_precision_score(yv[b], p0[b]): wu += 1
        r["boot_win_rate_auc"] = wa / 1000.0
        r["boot_win_rate_auprc"] = wu / 1000.0
        per_fold.append(r)
        print(f"  seed{seed} f{fold}: AUC {r['auc0']:.4f}->{r['auc6']:.4f} AUPRC {r['auprc0']:.4f}->{r['auprc6']:.4f} winAUC {r['boot_win_rate_auc']:.2f}", flush=True)

pf = {k: float(np.mean([f[k] for f in per_fold])) for k in
      ["auc0", "auc6", "auprc0", "auprc6", "acc0", "acc6", "boot_win_rate_auc", "boot_win_rate_auprc"]}
summary = {"per_fold": per_fold, "mean": pf,
           "n_folds_auc6_wins": int(sum(f["auc6"] > f["auc0"] for f in per_fold)),
           "n_folds_boot_auc_ge_95": int(sum(f["boot_win_rate_auc"] >= 0.95 for f in per_fold)),
           "n_folds_boot_auprc_ge_95": int(sum(f["boot_win_rate_auprc"] >= 0.95 for f in per_fold)),
           "auprc_not_degraded": bool(pf["auprc6"] >= pf["auprc0"] - 0.002)}
json.dump(summary, open(f"{AN}/groupcv_a0_vs_a6.json", "w"), indent=2)
print("summary:", json.dumps({k: v for k, v in summary.items() if k != "per_fold"}, indent=1), flush=True)
print(f"total {time.time()-t00:.0f}s ALL DONE", flush=True)
