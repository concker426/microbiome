#!/usr/bin/env python3
"""Ablation A0/A1/A2/A4 on unified qiita 2538 representation (clean_2538 split).
A0: SimpleEmb (genus emb) + mean pool + MLP
A1: A0 + rank-bin embedding (MGM-style rank encoding)
A2: A0 + abundance-bin embedding (quantile of within-sample abundance)
A4: A2 + masked-abundance pretraining (BiomeGPT-style), then fine-tune
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
P = "/hd/liujx/microbiome_llm_project"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E, B, SEEDS = 1224, 256, 32, [42, 123, 456]
EPOCHS = 50

# ---------- data ----------
qids = json.load(open(f"{P}/data/qiita_ibd/clean_2538/sample_ids.json"))
qv = np.load(f"{P}/data/qiita_ibd/clean_2538/vectors.npy")
qmap = {sid: i for i, sid in enumerate(qids)}
tr = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/train_nl.jsonl")]
te = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/test_nl.jsonl")]

def build(split):
    toks, rbin, abin, labs = [], [], [], []
    for d in split:
        sid = d["sample_id"]
        if sid not in qmap: continue
        v = qv[qmap[sid]]
        present = np.where(v > 0)[0]
        order = present[np.argsort(-v[present])]
        t = (order + 1).astype(np.int64)
        vals = v[order]
        N = len(t)
        rb = np.minimum(np.floor(np.arange(N) / max(N, 1) * B), B - 1).astype(np.int64)
        q = (np.argsort(np.argsort(vals)) / max(N - 1, 1))
        ab = np.minimum(np.floor(q * B), B - 1).astype(np.int64)
        toks.append(t); rbin.append(rb); abin.append(ab)
        labs.append(1 if d["label"] == "Disease" else 0)
    return toks, rbin, abin, np.array(labs)

tr_toks, tr_rb, tr_ab, tr_y = build(tr)
te_toks, te_rb, te_ab, te_y = build(te)
print(f"train {len(tr_toks)} (disease {tr_y.sum()}), test {len(te_toks)} (disease {te_y.sum()})")

def pad(seqs):
    ml = max(len(t) for t in seqs)
    X = np.zeros((len(seqs), ml), dtype=np.int64); M = np.zeros((len(seqs), ml), dtype=np.float32)
    for i, t in enumerate(seqs): X[i, :len(t)] = t; M[i, :len(t)] = 1
    return X, M

Xtr, Mtr = pad(tr_toks); Xte, Mte = pad(te_toks)
Rtr = pad(tr_rb)[0]; Rte = pad(te_rb)[0]
Atr = pad(tr_ab)[0]; Ate = pad(te_ab)[0]

# ---------- models ----------
class Enc(nn.Module):
    def __init__(self, use_rank=False, use_abund=False):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
        self.use_rank = use_rank; self.use_abund = use_abund
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
        self.fc1 = nn.Linear(E, 128); self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(128, 2)
    def forward(self, x): return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class Cls(nn.Module):
    def __init__(self, use_rank=False, use_abund=False):
        super().__init__()
        self.enc = Enc(use_rank, use_abund); self.mlp = MLP()
    def forward(self, ids, mask, rb=None, ab=None):
        return self.mlp(self.enc(ids, mask, rb, ab))

def evaluate(model, use_rank, use_abund):
    model.eval()
    with torch.no_grad():
        rb = torch.tensor(Rte).to(DEVICE) if use_rank else None
        ab = torch.tensor(Ate).to(DEVICE) if use_abund else None
        p = torch.softmax(model(torch.tensor(Xte).to(DEVICE), torch.tensor(Mte).to(DEVICE), rb, ab), 1)[:, 1].cpu().numpy()
    return accuracy_score(te_y, (p > 0.5).astype(int)), roc_auc_score(te_y, p), average_precision_score(te_y, p)

def train_cls(use_rank=False, use_abund=False, pretrained_enc=None):
    accs, aucs, aprs = [], [], []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        m = Cls(use_rank, use_abund).to(DEVICE)
        if pretrained_enc is not None:
            m.enc.emb.weight.data.copy_(pretrained_enc.emb.weight.data)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        y = torch.tensor(tr_y, dtype=torch.long).to(DEVICE)
        w = torch.tensor([1.0, 1.5]).to(DEVICE)
        rb_t = torch.tensor(Rtr).to(DEVICE) if use_rank else None
        ab_t = torch.tensor(Atr).to(DEVICE) if use_abund else None
        m.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                idx = perm[i:i+32]
                opt.zero_grad()
                logits = m(torch.tensor(Xtr[idx]).to(DEVICE), torch.tensor(Mtr[idx]).to(DEVICE),
                           rb_t[idx] if rb_t is not None else None, ab_t[idx] if ab_t is not None else None)
                loss = F.cross_entropy(logits, y[idx], weight=w)
                loss.backward(); opt.step()
        a, u, ap = evaluate(m, use_rank, use_abund)
        accs.append(a); aucs.append(u); aprs.append(ap)
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs), np.mean(aprs), np.std(aprs)

# ---------- run ablations ----------
res = {}
acc, asd, auc, usd, apr, psd = train_cls(False, False)
res["A0"] = {"acc": acc, "auc": auc, "auprc": apr}
print(f"A0 (baseline):        ACC {acc:.4f} +/- {asd:.4f} | AUROC {auc:.4f} +/- {usd:.4f} | AUPRC {apr:.4f} +/- {psd:.4f}")

acc, asd, auc, usd, apr, psd = train_cls(True, False)
res["A1"] = {"acc": acc, "auc": auc, "auprc": apr}
print(f"A1 (+rank-bin):       ACC {acc:.4f} +/- {asd:.4f} | AUROC {auc:.4f} +/- {usd:.4f} | AUPRC {apr:.4f} +/- {psd:.4f}")

acc, asd, auc, usd, apr, psd = train_cls(False, True)
res["A2"] = {"acc": acc, "auc": auc, "auprc": apr}
print(f"A2 (+abundance-bin):  ACC {acc:.4f} +/- {asd:.4f} | AUROC {auc:.4f} +/- {usd:.4f} | AUPRC {apr:.4f} +/- {psd:.4f}")

# A4: masked abundance pretraining (predict abundance bin of masked genera)
print("A4: masked-abundance pretraining ...")
def pretrain_masked():
    torch.manual_seed(42); np.random.seed(42)
    enc = Enc(False, True).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(30):
        enc.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            ids = torch.tensor(Xtr[idx]).to(DEVICE)
            mask = torch.tensor(Mtr[idx]).to(DEVICE)
            ab = torch.tensor(Atr[idx]).to(DEVICE)
            # randomly mask 25% of present tokens
            rm = (torch.rand(ids.shape, device=DEVICE) < 0.25).float() * mask
            # predict abundance bin of masked tokens via pooled context + genus embedding
            x = enc.emb(ids) + enc.abund_emb(ab.clamp(0, B-1))
            # context: mean of UNmasked
            um = mask - rm
            ctx = (x * um.unsqueeze(-1)).sum(1) / um.sum(1).clamp(min=1).unsqueeze(-1)
            # for each masked token, predict bin: logits = (genus_emb[i] * ctx).sum -> scalar per bin is too complex; use MLP head
            # simple reconstruction head
            logits = enc.abund_head(ctx)  # [B, B]
            # target: abundance bin of masked tokens (majority)
            # (simplified: predict the mean abundance-bin distribution)
            tgt = torch.zeros(len(idx), B, device=DEVICE)
            for b in range(len(idx)):
                mpos = (rm[b] > 0)
                if mpos.sum() > 0:
                    bins = ab[b][mpos].long().clamp(0, B-1)
                    tgt[b] = torch.bincount(bins, minlength=B).float() / bins.numel()
            loss = F.kl_div(F.log_softmax(logits, 1), tgt, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    return enc
# (A4 simplified pretrain implemented inline below with proper head)
class PretrainEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
        self.abund_emb = nn.Embedding(B, E)
        self.head = nn.Linear(E, B)
    def forward(self, ids, mask, ab, rm):
        x = self.emb(ids) + self.abund_emb(ab.clamp(0, B-1))
        um = mask - rm
        ctx = (x * um.unsqueeze(-1)).sum(1) / um.sum(1).clamp(min=1).unsqueeze(-1)
        return self.head(ctx)
def pretrain():
    torch.manual_seed(42); np.random.seed(42)
    m = PretrainEnc().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(30):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            ids = torch.tensor(Xtr[idx]).to(DEVICE); mask = torch.tensor(Mtr[idx]).to(DEVICE)
            ab = torch.tensor(Atr[idx]).to(DEVICE)
            rm = (torch.rand(ids.shape, device=DEVICE) < 0.25).float() * mask
            logits = m(ids, mask, ab, rm)
            tgt = torch.zeros(len(idx), B, device=DEVICE)
            for b in range(len(idx)):
                mpos = rm[b] > 0
                if mpos.sum() > 0:
                    bins = ab[b][mpos].long().clamp(0, B-1)
                    tgt[b] = torch.bincount(bins, minlength=B).float() / bins.numel()
            loss = F.kl_div(F.log_softmax(logits, 1), tgt, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return m
pt = pretrain()
accs, aucs, aprs = [], [], []
for sd in SEEDS:
    torch.manual_seed(sd); np.random.seed(sd)
    m = Cls(False, True).to(DEVICE)
    m.enc.emb.weight.data.copy_(pt.emb.weight.data)
    m.enc.abund_emb.weight.data.copy_(pt.abund_emb.weight.data)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.tensor(tr_y, dtype=torch.long).to(DEVICE); w = torch.tensor([1.0, 1.5]).to(DEVICE)
    ab_t = torch.tensor(Atr).to(DEVICE)
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]; opt.zero_grad()
            logits = m(torch.tensor(Xtr[idx]).to(DEVICE), torch.tensor(Mtr[idx]).to(DEVICE), None, ab_t[idx])
            loss = F.cross_entropy(logits, y[idx], weight=w); loss.backward(); opt.step()
    a, u, ap = evaluate(m, False, True)
    accs.append(a); aucs.append(u); aprs.append(ap)
res["A4"] = {"acc": np.mean(accs), "auc": np.mean(aucs), "auprc": np.mean(aprs)}
print(f"A4 (+masked pretrain): ACC {np.mean(accs):.4f} +/- {np.std(accs):.4f} | AUROC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | AUPRC {np.mean(aprs):.4f} +/- {np.std(aprs):.4f}")

json.dump(res, open(f"{P}/ProCyon_v2/analysis/ablation_a0_a1_a2_a4.json", "w"), indent=2)
print("Saved ablation_a0_a1_a2_a4.json")