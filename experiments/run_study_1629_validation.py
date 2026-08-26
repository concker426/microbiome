#!/usr/bin/env python3
"""External validation on Study 1629 (683 Crohn's samples) using the unified qiita representation."""
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

# train data: qiita 2538 with clean labels
qi = json.load(open(f"{P}/data/qiita_ibd/qiita_ibd_info.json"))
qids = qi["sample_ids"][0:907]
qv2538 = np.load(f"{P}/data/qiita_ibd/qiita_ibd_vectors.npy")[0:907]
qmap = {sid: i for i, sid in enumerate(qids)}
tr = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/train_nl.jsonl")]
te = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/test_nl.jsonl")]
def build(split, vecs, vmap):
    ids = [d["sample_id"] for d in split if d["sample_id"] in vmap]
    seqs = [ab_to_tok(vecs[vmap[s]]) for s in ids]
    labs = np.array([1 if d["label"] == "Disease" else 0 for d in split if d["sample_id"] in vmap])
    return seqs, labs
trs, trl = build(tr, qv2538, qmap)
tes, tel = build(te, qv2538, qmap)
Xtr, Mtr = pad(trs); Xte, Mte = pad(tes)

# external: study 1629
v1629 = np.load(f"{P}/data/study_1629_vectors.npy")
lab1629 = json.load(open(f"{P}/data/study_1629_labels.json"))
y1629 = np.array([1 if l in ("Disease", "IBD") else 0 for l in lab1629["labels"]])
seqs1629 = [ab_to_tok(v) for v in v1629]
Xe, Me = pad(seqs1629)
print(f"1629: n={len(y1629)} vectors={v1629.shape}")

# SimpleEmb ensemble
probs = []
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
    probs.append(p)
p_simple = np.mean(probs, axis=0)
pred_simple = (p_simple > 0.5).astype(int)
rec_simple = float((pred_simple[y1629 == 1] == 1).mean())
print(f"SimpleEmb on 1629: disease identification = {rec_simple:.4f} ({(pred_simple[y1629==1]==1).sum()}/{y1629.sum()})")

# trees
clfs = {
    "HGB": HistGradientBoostingClassifier(max_iter=200, random_state=42).fit(qv2538[[qmap[d["sample_id"]] for d in tr]], trl),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42).fit(qv2538[[qmap[d["sample_id"]] for d in tr]], trl),
}
for nm, c in clfs.items():
    p = c.predict_proba(v1629)[:, 1]
    pred = (p > 0.5).astype(int)
    rec = float((pred[y1629 == 1] == 1).mean())
    print(f"{nm} on 1629: disease identification = {rec:.4f} ({(pred[y1629==1]==1).sum()}/{y1629.sum()})")

out = {
    "1629": {
        "n": int(len(y1629)),
        "n_disease": int(y1629.sum()),
        "SimpleEmb_disease_identification": float(rec_simple),
        "HGB_disease_identification": float((clfs["HGB"].predict(v1629) == 1).mean()),
        "RF_disease_identification": float((clfs["RF"].predict(v1629) == 1).mean()),
        "SimpleEmb_probs": p_simple.tolist(),
    }
}
json.dump(out, open(f"{P}/ProCyon_v2/analysis/study_1629_external_validation.json", "w"), indent=2)
print("Saved study_1629_external_validation.json")
