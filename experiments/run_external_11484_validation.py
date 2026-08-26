#!/usr/bin/env python3
"""External validation: ProCyon v2 (SimpleEmb, E=768) on independent Study 11484 (96 IBD samples)."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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

model = Model()
sd = torch.load(f"{P}/ProCyon_v2/backbone/final_model.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
model.load_state_dict(sd)
model.to(DEVICE).eval()

def predict(seqs, masks):
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(seqs).to(DEVICE), torch.tensor(masks).to(DEVICE)), 1)[:, 1].cpu().numpy()
    return probs

def abundance_to_tokens(vec):
    present = np.where(vec > 0)[0]
    order = present[np.argsort(-vec[present])]
    return order + 1

def pad_batch(tok_lists, max_len=None):
    if max_len is None: max_len = max(len(t) for t in tok_lists)
    Xs = np.zeros((len(tok_lists), max_len), dtype=np.int64)
    Xm = np.zeros((len(tok_lists), max_len), dtype=np.float32)
    for i, t in enumerate(tok_lists):
        Xs[i, :len(t)] = t; Xm[i, :len(t)] = 1
    return Xs, Xm

out = {}

# ---- Sanity: clean_2538 held-out test ----
xs = np.load(f"{P}/data/qiita_ibd/clean_2538/test_genus_sequences.npy")
xm = np.load(f"{P}/data/qiita_ibd/clean_2538/test_genus_masks.npy")
nl = [json.loads(l) for l in open(f"{P}/data/qiita_ibd/clean_2538/test_nl.jsonl")]
y_clean = np.array([1 if d["label"] == "Disease" else 0 for d in nl])
p_clean = predict(xs, xm)
acc_clean = ((p_clean > 0.5).astype(int) == y_clean).mean()
print(f"Sanity check clean_2538 test: ACC={acc_clean:.4f} (expect ~0.9257)")
out["clean_test_acc"] = float(acc_clean)

# ---- External: Study 11484 ----
vecs = np.load(f"{P}/data/study_11484_vectors.npy")
lab = json.load(open(f"{P}/data/study_11484_labels.json"))
labels = np.array([1 if l == "Disease" else 0 for l in lab["labels"]])
ids = lab["ids"]
print(f"11484: shape={vecs.shape}, IBD={labels.sum()}, Healthy={len(labels)-labels.sum()}")
tok_lists = [abundance_to_tokens(v) for v in vecs]
Xs, Xm = pad_batch(tok_lists)
probs = predict(Xs, Xm)
preds = (probs > 0.5).astype(int)
n_ibd = int(labels.sum())
disease_recall = float((preds[labels == 1] == 1).sum()) / n_ibd if n_ibd else 0.0
acc_ext = float((preds == labels).mean())
print(f"External 11484: Disease identification = {disease_recall:.4f} ({(preds[labels==1]==1).sum()}/{n_ibd}), ACC={acc_ext:.4f}")
out["external"] = {"n": int(len(labels)), "n_ibd": n_ibd,
    "disease_recall": disease_recall, "acc": acc_ext,
    "probs": probs.tolist(), "preds": preds.tolist(), "labels": labels.tolist(), "ids": ids}

# ---- Tree baseline (HGB/RF) trained on clean_2538 raw abundance, tested on 11484 ----
try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    clean_vecs = np.load(f"{P}/data/qiita_ibd/clean_2538/vectors.npy")
    clean_lab = np.load(f"{P}/data/qiita_ibd/clean_2538/labels.npy", allow_pickle=True)
    if clean_lab.dtype != np.int64 and clean_lab.ndim == 2: clean_lab = clean_lab.ravel()
    # align feature dims to 1223 (combined vocab) if needed
    if clean_vecs.shape[1] == 1226: clean_vecs = clean_vecs[:, :1223]
    print(f"clean vectors: {clean_vecs.shape}, labels: {clean_lab.shape} {clean_lab[:3]}")
    ext_vecs = vecs[:, :clean_vecs.shape[1]]
    for name, clf in [("HGB", HistGradientBoostingClassifier(max_iter=200)),
                      ("RF", RandomForestClassifier(n_estimators=200, random_state=42))]:
        clf.fit(clean_vecs, clean_lab)
        p = clf.predict(ext_vecs)
        rec = float((p[labels == 1] == 1).sum()) / n_ibd if n_ibd else 0.0
        accm = float((p == labels).mean())
        print(f"{name} external: Disease ident={rec:.4f}, ACC={accm:.4f}")
        out[f"{name.lower()}_external"] = {"disease_recall": rec, "acc": accm}
except Exception as e:
    print("Tree baseline skipped:", e)

json.dump(out, open(f"{P}/ProCyon_v2/analysis/external_11484_validation.json", "w"), indent=2)
print("Saved: ProCyon_v2/analysis/external_11484_validation.json")
