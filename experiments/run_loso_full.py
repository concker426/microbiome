#!/usr/bin/env python3
"""Leave-one-source-out (LOSO) external validation on unified qiita/combined vector representation.
Train: study_2538 labeled samples (clean_2538 split, 659 train / 167 test), rank-tokenized.
External: 10317 (194 IBD + 388 Healthy), 10283 (568 IBD), 1939 (1359 IBD), 11484 (96 IBD).
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
P = "/hd/liujx/microbiome_llm_project"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E = 1224, 768
SEEDS = [42, 123, 456]

class Emb(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(self, ids, mask):
        x = self.emb(ids); mf = mask.float().unsqueeze(-1)
        return (x * mf).sum(1) / mf.sum(1).clamp(min=1)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(E, 256); self.bn1 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Emb(); self.mlp = MLP()
    def forward(self, ids, mask):
        return self.mlp(self.enc(ids, mask))

def ab_to_tok(v):
    pres = np.where(v > 0)[0]
    return pres[np.argsort(-v[pres])] + 1
def pad(seqs):
    ml = max(len(t) for t in seqs)
    Xs = np.zeros((len(seqs), ml), dtype=np.int64); Xm = np.zeros((len(seqs), ml), dtype=np.float32)
    for i, t in enumerate(seqs):
        Xs[i, :len(t)] = t; Xm[i, :len(t)] = 1
    return Xs, Xm

# ---------- train data (study_2538, qiita repr) ----------
qi = json.load(open(f"{P}/data/qiita_ibd/qiita_ibd_info.json"))
qids = qi["sample_ids"][0:907]
qv2538 = np.load(f"{P}/data/qiita_ibd/qiita_ibd_vectors.npy")[0:907]
qmap = {sid: i for i, sid in enumerate(qids)}
tr = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/train_nl.jsonl")]
te = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/test_nl.jsonl")]
def build(split, vecs, vmap):
    ids = [d["sample_id"] for d in split if d["sample_id"] in vmap]
    seqs = [ab_to_tok(vecs[vmap[s]]) for s in ids]
    labs = [1 if d["label"] == "Disease" else 0 for d in split if d["sample_id"] in vmap]
    return seqs, np.array(labs), ids
trs, trl, _ = build(tr, qv2538, qmap)
tes, tel, _ = build(te, qv2538, qmap)
Xtr, Mtr = pad(trs); Xte, Mte = pad(tes)
print(f"train {len(trs)} (disease {trl.sum()}) | test {len(tes)} (disease {tel.sum()})")

# ---------- external datasets ----------
info = json.load(open(f"{P}/data/qiita_ibd/combined_info.json"))
cids, clabs = info["sample_ids"], info["labels"]
cv = np.load(f"{P}/data/qiita_ibd/combined_vectors.npy")
qv = np.load(f"{P}/data/qiita_ibd/qiita_ibd_vectors.npy")
v114 = np.load(f"{P}/data/study_11484_vectors.npy")
def ext_10317():
    seqs = [ab_to_tok(cv[i]) for i in range(582)]
    labs = np.array([1 if clabs[i] in ("IBD", "Disease") else 0 for i in range(582)])
    return seqs, labs
def ext_block(a, b, lab):
    seqs = [ab_to_tok(qv[i]) for i in range(a, b)]
    labs = np.array([lab] * (b - a))
    return seqs, labs
ext = {
    "10317": ext_10317(),
    "10283": ext_block(907, 1475, 1),
    "1939": ext_block(1475, 2834, 1),
    "11484": ([ab_to_tok(v) for v in v114], np.ones(len(v114), dtype=int)),
}

# ---------- trees ----------
clfs = {nm: c.fit(qv2538[[qmap[d["sample_id"]] for d in tr]], trl)
        for nm, c in [("HGB", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
                       ("RF", RandomForestClassifier(n_estimators=200, random_state=42))]}
tree_data = {"10317": cv[0:582], "10283": qv[907:1475], "1939": qv[1475:2834], "11484": v114}

def eval_metrics(labs, probs):
    y = np.array(labs); pr = np.array(probs)
    pred = (pr > 0.5).astype(int)
    r = {"n": int(len(y)), "n_disease": int(y.sum())}
    r["disease_recall"] = float((pred[y == 1] == 1).sum()) / max(int(y.sum()), 1)
    r["acc"] = float((pred == y).mean())
    if len(set(y)) == 2:
        r["auc"] = float(roc_auc_score(y, pr))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        r["sensitivity"] = float(tp) / max(tp + fn, 1)
        r["specificity"] = float(tn) / max(tn + fp, 1)
    return r

out = {}
for name, (seqs, labs) in ext.items():
    Xe, Me = pad(seqs)
    # SimpleEmb 3-seed ensemble
    probs_list = []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        model = Model().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ytr = torch.tensor(trl, dtype=torch.long).to(DEVICE)
        w = torch.tensor([1.0, 1.5]).to(DEVICE)
        model.train()
        for ep in range(50):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                idx = perm[i:i+32]
                opt.zero_grad()
                logits = model(torch.tensor(Xtr[idx]).to(DEVICE), torch.tensor(Mtr[idx]).to(DEVICE))
                loss = F.cross_entropy(logits, ytr[idx], weight=w)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(Xe).to(DEVICE), torch.tensor(Me).to(DEVICE)), 1)[:, 1].cpu().numpy()
        probs_list.append(p)
    p_ens = np.mean(probs_list, axis=0)
    r_simple = eval_metrics(labs, p_ens)
    # internal reference: test on 2538 holdout
    if name == "10317":
        with torch.no_grad():
            pi = torch.softmax(model(torch.tensor(Xte).to(DEVICE), torch.tensor(Mte).to(DEVICE)), 1)[:, 1].cpu().numpy()
        out["2538_holdout"] = eval_metrics(tel, pi)
    # trees
    r_trees = {}
    for nm, c in clfs.items():
        tp_ = c.predict_proba(tree_data[name])[:, 1]
        r_trees[nm] = eval_metrics(labs, tp_)
    out[name] = {"SimpleEmb": r_simple, "trees": r_trees}
    def f(r):
        s = f"  n={r[chr(110)]} dis={r[chr(110)+chr(95)+chr(100)+chr(105)+chr(115)+chr(101)+chr(97)+chr(115)+chr(101)]} rec={r[chr(100)+chr(105)+chr(115)+chr(101)+chr(97)+chr(115)+chr(101)+chr(95)+chr(114)+chr(101)+chr(99)+chr(97)+chr(108)+chr(108)]:.3f} acc={r[chr(97)+chr(99)+chr(99)]:.3f}"
        return s + (f" auc={r[chr(97)+chr(117)+chr(99)]:.3f}" if "auc" in r else "")
    print(f"[{name}] SimpleEmb:{f(r_simple)}")
    for nm, r in r_trees.items(): print(f"[{name}] {nm}:{f(r)}")
json.dump(out, open(f"{P}/ProCyon_v2/analysis/loso_external_validation.json", "w"), indent=2)
print("Saved loso_external_validation.json")
