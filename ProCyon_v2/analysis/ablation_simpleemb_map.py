#!/usr/bin/env python3
"""SimpleEmb-MAP ablation A0/A1/A2/A4 on clean_2538 holdout (paper convention).

A0: SimpleEmb (nn.Embedding(1226,768)) + masked mean pool + MLP(768->256->2)   [paper baseline]
A1: A0 + rank-bin embedding (MGM-style rank encoding, B=32)
A2: A0 + abundance-bin embedding (within-sample quantile bin, B=32)
A4: A2 + masked-abundance pretraining (BiomeGPT-style), then fine-tune

Convention matches experiments/run_week1_experiments.py:
  labels from *_nl.jsonl (Disease=1), sample weight 1.5 for Disease,
  50 epochs, BS=32, AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR,
  SEEDS=[42,123,456,789,1024], metrics = mean over seeds.
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

BASE = "/hd/liujx/microbiome_llm_project/data/qiita_ibd"
C = f"{BASE}/clean_2538"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E, HID, B = 1226, 768, 256, 32
SEEDS = [42, 123, 456, 789, 1024]
NE, BS, LR, WD = 50, 32, 1e-3, 1e-4

# ---------- data ----------
vocab = json.load(open("/hd/liujx/microbiome_data/qiita_pretrain/genus_vocab.json"))
n2t = {k[3:]: v for k, v in vocab.items() if k.startswith("g__")}
qi = json.load(open(f"{BASE}/qiita_ibd_info.json"))
gnames = qi["genus_names"]
sv = np.load(f"{BASE}/study_2538_vectors.npy")
sv_ids = json.load(open(f"{BASE}/study_2538_ids.json"))
sv_pos = {s: i for i, s in enumerate(sv_ids)}

def load_split(name):
    data = [json.loads(l) for l in open(f"{C}/{name}_nl.jsonl")]
    seqs = np.load(f"{C}/{name}_genus_sequences.npy")
    masks = np.load(f"{C}/{name}_genus_masks.npy")
    y = np.array([1 if d["label"] == "Disease" else 0 for d in data])
    w = np.array([1.5 if d["label"] == "Disease" else 1.0 for d in data])
    # per-token abundance: map genus token -> abundance in the sample's study vector
    abund_seqs = np.zeros_like(seqs, dtype=np.float32)
    for j, d in enumerate(data):
        row = sv[sv_pos[d["sample_id"]]]
        tok2val = {}
        for g in np.where(row > 0)[0]:
            t = n2t.get(gnames[g])
            if t is not None and t < V:
                tok2val[t] = row[g]
        for k, t in enumerate(seqs[j]):
            if masks[j, k]:
                abund_seqs[j, k] = tok2val.get(int(t), 0.0)
    return seqs, masks, y, w, abund_seqs

tr_s, tr_m, tr_y, tr_w, tr_a = load_split("train")
te_s, te_m, te_y, te_w, te_a = load_split("test")
print(f"train {len(tr_y)} (D={tr_y.sum()}), test {len(te_y)} (D={te_y.sum()})")

def bins(seqs, masks, abund):
    """rank-bin (position among present tokens) and abundance-bin (within-sample quantile)."""
    n = len(seqs)
    rb = np.zeros_like(seqs, dtype=np.int64)
    ab = np.zeros_like(seqs, dtype=np.int64)
    for j in range(n):
        pos = np.where(masks[j])[0]
        N = len(pos)
        if N == 0:
            continue
        rb[j, pos] = np.minimum((np.arange(N) / N * B).astype(np.int64), B - 1)
        vals = abund[j, pos]
        q = np.argsort(np.argsort(vals)) / max(N - 1, 1)
        ab[j, pos] = np.minimum((q * B).astype(np.int64), B - 1)
    return rb, ab

tr_rb, tr_ab = bins(tr_s, tr_m, tr_a)
te_rb, te_ab = bins(te_s, te_m, te_a)

def t(x):
    return torch.tensor(x).long().to(DEVICE)

Xtr, Mtr = t(tr_s), t(tr_m); Xte, Mte = t(te_s), t(te_m)
Rtr, Atr = t(tr_rb), t(tr_ab); Rte, Ate = t(te_rb), t(te_ab)
Ytr = torch.tensor(tr_y).long().to(DEVICE)
Wtr = torch.tensor(tr_w).float().to(DEVICE)

# ---------- models ----------
class Enc(nn.Module):
    def __init__(self, use_rank=False, use_abund=False):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
        self.use_rank, self.use_abund = use_rank, use_abund
        if use_rank: self.rank_emb = nn.Embedding(B, E)
        if use_abund: self.abund_emb = nn.Embedding(B, E)
    def forward(self, ids, mask, rb=None, ab=None):
        x = self.emb(ids)
        if self.use_rank: x = x + self.rank_emb(rb)
        if self.use_abund: x = x + self.abund_emb(ab)
        mf = mask.float().unsqueeze(-1)
        return (x * mf).sum(1) / mf.sum(1).clamp(min=1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(E, HID); self.bn1 = nn.BatchNorm1d(HID)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(HID, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class Cls(nn.Module):
    def __init__(self, use_rank=False, use_abund=False):
        super().__init__()
        self.enc = Enc(use_rank, use_abund); self.mlp = MLP()
    def forward(self, ids, mask, rb=None, ab=None):
        return self.mlp(self.enc(ids, mask, rb, ab))

def evaluate(m, use_rank, use_abund):
    m.eval()
    with torch.no_grad():
        rb, ab = (Rte if use_rank else None), (Ate if use_abund else None)
        p = torch.softmax(m(Xte, Mte, rb, ab), 1)[:, 1].cpu().numpy()
    return (accuracy_score(te_y, (p > 0.5).astype(int)),
            roc_auc_score(te_y, p), average_precision_score(te_y, p))

def train_cls(use_rank=False, use_abund=False, pretr=None):
    mets = []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        m = Cls(use_rank, use_abund).to(DEVICE)
        if pretr is not None:
            m.enc.emb.weight.data.copy_(pretr["emb"])
            if use_abund: m.enc.abund_emb.weight.data.copy_(pretr["abund"])
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
        rb = Rtr if use_rank else None; ab = Atr if use_abund else None
        for ep in range(NE):
            m.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), BS):
                idx = perm[i:i + BS]
                if len(idx) < 2: continue  # BatchNorm needs >1 sample
                logits = m(Xtr[idx], Mtr[idx], rb[idx] if rb is not None else None,
                           ab[idx] if ab is not None else None)
                loss = (F.cross_entropy(logits, Ytr[idx], reduction="none") * Wtr[idx]).sum() / Wtr[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        mets.append(evaluate(m, use_rank, use_abund))
    a = np.array(mets)
    return {"acc": a[:, 0].mean(), "acc_std": a[:, 0].std(),
            "auc": a[:, 1].mean(), "auc_std": a[:, 1].std(),
            "auprc": a[:, 2].mean(), "auprc_std": a[:, 2].std()}

# ---------- A4 pretraining: masked abundance-bin prediction ----------
def pretrain_masked():
    torch.manual_seed(42); np.random.seed(42)
    emb = nn.Embedding(V, E, padding_idx=0).to(DEVICE)
    abe = nn.Embedding(B, E).to(DEVICE)
    head = nn.Linear(2 * E, B).to(DEVICE)
    params = list(emb.parameters()) + list(abe.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    for ep in range(30):
        perm = torch.randperm(len(Xtr))
        tot = 0.0; nb = 0
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            if len(idx) < 2: continue
            ids, mask, ab = Xtr[idx], Mtr[idx], Atr[idx]
            real = (mask.bool()) & (ids != 3)  # present, non-EOS tokens
            rm = (torch.rand(ids.shape, device=DEVICE) < 0.25) & real  # masked set
            x = emb(ids) + abe(ab)
            mf = (mask.bool() & ~rm).float().unsqueeze(-1)
            ctx = (x * mf).sum(1) / mf.sum(1).clamp(min=1)  # [b, E]
            sel = rm.nonzero(as_tuple=False)  # positions to predict
            if len(sel) == 0: continue
            g = emb(ids[sel[:, 0], sel[:, 1]])  # [n, E]
            logits = head(torch.cat([g, ctx[sel[:, 0]]], 1))  # [n, B]
            tgt = ab[sel[:, 0], sel[:, 1]]
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == 29:
            print(f"  pretrain ep{ep}: loss {tot / max(nb,1):.4f}")
    return {"emb": emb.weight.data.clone(), "abund": abe.weight.data.clone()}

# ---------- run ----------
res = {}
for name, ur, ua, pt in [("A0", False, False, None),
                         ("A1", True, False, None),
                         ("A2", False, True, None)]:
    r = train_cls(ur, ua, pt)
    res[name] = r
    print(f"{name}: ACC {r['acc']:.4f}±{r['acc_std']:.4f} | AUROC {r['auc']:.4f}±{r['auc_std']:.4f} | AUPRC {r['auprc']:.4f}±{r['auprc_std']:.4f}", flush=True)

print("A4: masked-abundance pretraining ...", flush=True)
pt = pretrain_masked()
r = train_cls(False, True, pt)
res["A4"] = r
print(f"A4: ACC {r['acc']:.4f}±{r['acc_std']:.4f} | AUROC {r['auc']:.4f}±{r['auc_std']:.4f} | AUPRC {r['auprc']:.4f}±{r['auprc_std']:.4f}", flush=True)

out = "/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/ablation_simpleemb_map.json"
json.dump(res, open(out, "w"), indent=2)
print("Saved", out)
