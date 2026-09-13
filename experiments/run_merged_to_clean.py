import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
P = "/hd/liujx/microbiome_llm_project"; DEVICE = "cuda:0"
V, E, BS, LR_RATE, WD, NE = 1226, 768, 32, 1e-3, 1e-4, 50
class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0)
    def forward(self, ids, mask=None):
        x = self.emb(ids)
        mf = mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return (x*mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
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
class DS(Dataset):
    def __init__(self, data, seqs, masks):
        self.seqs = seqs; self.masks = masks
        self.labels = np.array([1 if d["label"]=="Disease" else 0 for d in data])
        self.sw = np.array([1.5 if d.get("label","Healthy")=="Disease" else 1.0 for d in data])
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i].astype(np.int64)), torch.tensor(self.masks[i], dtype=torch.bool), torch.tensor(self.labels[i]), torch.tensor(self.sw[i]))
def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
    y=torch.stack([x[2] for x in batch]); sw=torch.stack([x[3] for x in batch])
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y,sw
def load_ds(name):
    p = f"{P}/data/qiita_ibd/{name}"
    tr = [json.loads(l) for l in open(f"{p}/train_nl.jsonl")]
    te = [json.loads(l) for l in open(f"{p}/test_nl.jsonl")]
    ts = np.load(f"{p}/train_genus_sequences.npy"); xs = np.load(f"{p}/test_genus_sequences.npy")
    tm = np.load(f"{p}/train_genus_masks.npy"); xm = np.load(f"{p}/test_genus_masks.npy")
    return tr, te, ts, xs, tm, xm
mtr, mte, mts, mxs, mtm, mxm = load_ds("merged_all")
ctr, cte, cts, cxs, ctm, cxm = load_ds("clean_2538")
torch.manual_seed(42); np.random.seed(42)
train_ds = DS(mtr, mts, mtm)
test_ds = DS(cte, cxs, cxm)
tl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
el = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)
model = Model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
for ep in range(NE):
    model.train()
    for gi, gm, y, sw in tl:
        gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
        loss = (F.cross_entropy(model(gi, gm), y, reduction="none") * sw).sum() / sw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
model.eval(); probs = []
with torch.no_grad():
    for gi, gm, y, sw in el:
        gi, gm = gi.to(DEVICE), gm.to(DEVICE)
        probs.append(F.softmax(model(gi, gm), dim=1)[:, 1].cpu().numpy())
probs = np.concatenate(probs)
y = np.array([1 if d["label"]=="Disease" else 0 for d in cte])
preds = (probs > 0.5).astype(int)
acc = accuracy_score(y, preds); auc = roc_auc_score(y, probs)
tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
print(f"merged->clean: ACC={acc:.4f} AUC={auc:.4f} Sens={tp/(tp+fn):.4f} Spec={tn/(tn+fp):.4f}")
json.dump({"acc": acc, "auc": auc}, open(f"{P}/ProCyon_v2/analysis/merged_to_clean_check.json", "w"), indent=2)
