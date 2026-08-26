#!/usr/bin/env python3
"""Leave-one-source-out external validation: SimpleEmb + trees on 3 external cohorts.
External cohorts (from combined/qiita vectors, 1223-genus vocab):
  - 10317 (external_ibd): 194 IBD + 388 Healthy  (balanced, with controls)
  - 10283: 568 IBD (disease-only)
  - 1939: 1359 IBD (disease-only)
Reference: Study 11484 (96 IBD) from earlier experiment.
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
P = "/hd/liujx/microbiome_llm_project"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
V, E = 1226, 768

class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[..., :1])
        return (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(E, 256); self.bn1 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3); self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn1(self.fc1(x)))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SimpleEmbEnc(); self.mlp = MLPHead()
    def forward(self, ids, mask=None):
        return self.mlp(self.enc(ids, mask))

def abundance_to_tokens(vec):
    present = np.where(vec > 0)[0]
    order = present[np.argsort(-vec[present])]
    return order + 1

def pad_batch(tok_lists):
    ml = max(len(t) for t in tok_lists)
    Xs = np.zeros((len(tok_lists), ml), dtype=np.int64)
    Xm = np.zeros((len(tok_lists), ml), dtype=np.float32)
    for i, t in enumerate(tok_lists):
        Xs[i, :len(t)] = t; Xm[i, :len(t)] = 1
    return Xs, Xm

def predict_probs(model, tok_lists):
    Xs, Xm = pad_batch(tok_lists)
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xs).to(DEVICE), torch.tensor(Xm).to(DEVICE)), 1)[:, 1].cpu().numpy()

def evaluate(name, labels, probs, has_controls=True):
    y = np.array([1 if l in ("Disease", "IBD") else 0 for l in labels])
    pred = (probs > 0.5).astype(int)
    res = {"n": int(len(y)), "n_disease": int(y.sum()), "n_healthy": int((1 - y).sum())}
    res["disease_recall"] = float((pred[y == 1] == 1).sum()) / max(y.sum(), 1)
    res["acc"] = float((pred == y).mean())
    if has_controls and len(np.unique(y)) == 2:
        res["auc"] = float(roc_auc_score(y, probs))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        res["sensitivity"] = float(tp) / max(tp + fn, 1)
        res["specificity"] = float(tn) / max(tn + fp, 1)
    print(f"{name}: n={res[chr(110)]} disease={res[chr(110)+chr(95)+chr(100)+chr(105)+chr(115)+chr(101)+chr(97)+chr(115)+chr(101)]} "
          f"recall={res[chr(100)+chr(105)+chr(115)+chr(101)+chr(97)+chr(115)+chr(101)+chr(95)+chr(114)+chr(101)+chr(99)+chr(97)+chr(108)+chr(108)]:.4f} acc={res[chr(97)+chr(99)+chr(99)]:.4f}"
          + (f" auc={res[chr(97)+chr(117)+chr(99)]:.4f} sens={res[chr(115)+chr(101)+chr(110)+chr(115)+chr(105)+chr(116)+chr(105)+chr(118)+chr(105)+chr(116)+chr(121)]:.4f} spec={res[chr(115)+chr(112)+chr(101)+chr(99)+chr(105)+chr(102)+chr(105)+chr(99)+chr(105)+chr(116)+chr(121)]:.4f}" if "auc" in res else ""))
    return res

# ---- load SimpleEmb ----
model = Model()
sd = torch.load(f"{P}/ProCyon_v2/backbone/final_model.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
model.load_state_dict(sd)
model.to(DEVICE).eval()

# ---- external datasets ----
info = json.load(open(f"{P}/data/qiita_ibd/combined_info.json"))
ids, labs, srcs = info["sample_ids"], info["labels"], info["sources"]
qi = json.load(open(f"{P}/data/qiita_ibd/qiita_ibd_info.json"))
qids = qi["sample_ids"]
qv = np.load(f"{P}/data/qiita_ibd/qiita_ibd_vectors.npy")
cv = np.load(f"{P}/data/qiita_ibd/combined_vectors.npy")

datasets = {}
datasets["10317"] = {"vecs": cv[0:582], "labels": labs[0:582]}
datasets["10283"] = {"vecs": qv[907:1475], "labels": [labs[ids.index(q) if q in ids else 0] for q in qids[907:1475]]}
datasets["1939"] = {"vecs": qv[1475:2834], "labels": [labs[ids.index(q) if q in ids else 0] for q in qids[1475:2834]]}
v114 = np.load(f"{P}/data/study_11484_vectors.npy")
l114 = json.load(open(f"{P}/data/study_11484_labels.json"))["labels"]
datasets["11484"] = {"vecs": v114, "labels": l114}

# ---- trees trained on clean_2538 raw abundance ----
clean_vecs = np.load(f"{P}/data/qiita_ibd/clean_2538/vectors.npy")
clean_lab = np.load(f"{P}/data/qiita_ibd/clean_2538/labels.npy", allow_pickle=True)
if clean_lab.ndim == 2: clean_lab = clean_lab.ravel()
clfs = {}
for nm, clf in [("HGB", HistGradientBoostingClassifier(max_iter=200)),
                ("RF", RandomForestClassifier(n_estimators=200, random_state=42))]:
    clf.fit(clean_vecs, clean_lab)
    clfs[nm] = clf

out = {}
for name, ds in datasets.items():
    vecs = ds["vecs"]
    labels = ds["labels"]
    has_ctrl = any(l in ("Healthy",) for l in labels)
    # SimpleEmb
    toks = [abundance_to_tokens(v) for v in vecs]
    probs = predict_probs(model, toks)
    r_simple = evaluate(f"[{name}] SimpleEmb", labels, probs, has_ctrl)
    # trees
    r_trees = {}
    for nm, clf in clfs.items():
        p = clf.predict_proba(vecs[:, :clean_vecs.shape[1]])[:, 1]
        r_trees[nm] = evaluate(f"[{name}] {nm}", labels, p, has_ctrl)
    out[name] = {"SimpleEmb": r_simple, "trees": r_trees}

json.dump(out, open(f"{P}/ProCyon_v2/analysis/loso_external_validation.json", "w"), indent=2)
print("Saved: ProCyon_v2/analysis/loso_external_validation.json")
