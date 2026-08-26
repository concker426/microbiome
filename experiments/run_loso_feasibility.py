import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
P = "/hd/liujx/microbiome_llm_project"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E = 1223, 256
class Emb(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, E, padding_idx=0)
    def forward(self, ids, mask):
        x = self.emb(ids); mf = mask.float().unsqueeze(-1)
        return (x * mf).sum(1) / mf.sum(1).clamp(min=1)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(E, 128); self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(128, 2)
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
qi = json.load(open(f"{P}/data/qiita_ibd/qiita_ibd_info.json"))
qids = qi["sample_ids"][0:907]
qv = np.load(f"{P}/data/qiita_ibd/qiita_ibd_vectors.npy")[0:907]
tr = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/train_nl.jsonl")]
te = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/test_nl.jsonl")]
qmap = {sid: i for i, sid in enumerate(qids)}
def build(split):
    ids = [d["sample_id"] for d in split]
    seqs = [ab_to_tok(qv[qmap[s]]) for s in ids if s in qmap]
    labs = [1 if d["label"] == "Disease" else 0 for d in split if d["sample_id"] in qmap]
    return seqs, labs
trs, trl = build(tr); tes, tel = build(te)
print("train:", len(trs), "test:", len(tes))
Xtr, Mtr = pad(trs); Xte, Mte = pad(tes)
model = Model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
ytr = torch.tensor(trl, dtype=torch.long).to(DEVICE)
w = torch.tensor([1.0, 1.5]).to(DEVICE)
model.train()
for ep in range(30):
    perm = torch.randperm(len(Xtr))
    tot = 0
    for i in range(0, len(Xtr), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(torch.tensor(Xtr[idx]).to(DEVICE), torch.tensor(Mtr[idx]).to(DEVICE))
        loss = F.cross_entropy(logits, ytr[idx], weight=w)
        loss.backward(); opt.step(); tot += loss.item()
    if ep % 10 == 9: print("ep", ep, "loss %.4f" % (tot / max(len(Xtr)//32, 1)))
model.eval()
with torch.no_grad():
    p = torch.softmax(model(torch.tensor(Xte).to(DEVICE), torch.tensor(Mte).to(DEVICE)), 1)[:, 1].cpu().numpy()
yte = np.array(tel)
acc = ((p > 0.5).astype(int) == yte).mean()
auc = roc_auc_score(yte, p)
rec = ((p[yte == 1] > 0.5)).mean()
print(f"LOSO feasibility (qiita-repr 2538): test n={len(yte)} ACC={acc:.4f} AUC={auc:.4f} recall={rec:.4f}")
