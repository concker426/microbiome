import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
P = "/hd/liujx/microbiome_llm_project"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E = 1226, 768
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
model = Model()
sd = torch.load(f"{P}/ProCyon_v2/backbone/final_model.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
model.load_state_dict(sd); model.to(DEVICE).eval()
mtest = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/merged_all/test_nl.jsonl")]
mxs = np.load(f"{P}/data/qiita_ibd/merged_all/test_genus_sequences.npy")
mxm = np.load(f"{P}/data/qiita_ibd/merged_all/test_genus_masks.npy")
with torch.no_grad():
    probs = torch.softmax(model(torch.tensor(mxs).to(DEVICE), torch.tensor(mxm).to(DEVICE)), 1)[:, 1].cpu().numpy()
by = {}
for i, d in enumerate(mtest):
    p = str(d["sample_id"]).split(".")[0]
    by.setdefault(p, []).append((d["label"], probs[i]))
for p, arr in sorted(by.items()):
    y = np.array([1 if l == "Disease" else 0 for l, _ in arr])
    pr = np.array([v for _, v in arr])
    acc = ((pr > 0.5).astype(int) == y).mean()
    rec = ((pr[y == 1] > 0.5)).mean() if y.sum() else float("nan")
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, pr) if len(set(y)) == 2 else None
    except: pass
    print(f"{p}: n={len(arr)} disease={y.sum()} acc={acc:.4f} recall={rec:.4f}" + (f" auc={auc:.4f}" if auc else ""))
