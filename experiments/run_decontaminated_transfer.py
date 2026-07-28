#!/usr/bin/env python3
"""Decontaminate and recompute clean->merged transfer"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

V=1226; E=768; BS=32; LR_RATE=1e-3; WD=1e-4; NE=50; DEVICE='cuda:0'

# Load data
def load_ds(name):
    p = f'/hd/liujx/microbiome_llm_project/data/qiita_ibd/{name}'
    tr = [json.loads(l) for l in open(f'{p}/train_nl.jsonl')]
    te = [json.loads(l) for l in open(f'{p}/test_nl.jsonl')]
    ts = np.load(f'{p}/train_genus_sequences.npy')
    xs = np.load(f'{p}/test_genus_sequences.npy')
    tm = np.load(f'{p}/train_genus_masks.npy')
    xm = np.load(f'{p}/test_genus_masks.npy')
    return tr, te, ts, xs, tm, xm

clean_tr, clean_te, cts, cxs, ctm, cxm = load_ds('clean_2538')
merged_tr, merged_te, mts, mxs, mtm, mxm = load_ds('merged_all')

# Decontaminate
clean_train_ids = set(d['sample_id'] for d in clean_tr)
keep = [i for i, d in enumerate(merged_te) if d['sample_id'] not in clean_train_ids]
print(f"Decontaminated: {len(keep)}/{len(merged_te)} samples retained (removed {len(merged_te)-len(keep)})")

mxs_clean = mxs[keep]; mxm_clean = mxm[keep]
merged_te_clean = [merged_te[i] for i in keep]
merged_labels = np.array([1 if d['label']=='Disease' else 0 for d in merged_te_clean])
print(f"Disease: {merged_labels.sum()}, Healthy: {len(merged_labels)-merged_labels.sum()}")

# Model
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
        self.labels = np.array([1 if d['label']=='Disease' else 0 for d in data])
        self.sw = np.array([1.5 if d.get('label','Healthy')=='Disease' else 1.0 for d in data])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                torch.tensor(self.masks[i],dtype=torch.bool),
                torch.tensor(self.labels[i],dtype=torch.long),
                torch.tensor(self.sw[i],dtype=torch.float32))

def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
    y=torch.stack([x[2] for x in batch]); sw=torch.stack([x[3] for x in batch])
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y,sw

# Train on clean, test on decontaminated merged
torch.manual_seed(42); np.random.seed(42)
train_ds = DS(clean_tr, cts, ctm)
test_ds = DS(merged_te_clean, mxs_clean, mxm_clean)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate)

model = Model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE)
for ep in range(NE):
    model.train()
    for gi, gm, y, sw in train_loader:
        gi, gm, y, sw = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE), sw.to(DEVICE)
        loss = (F.cross_entropy(model(gi, gm), y, reduction='none') * sw).sum() / sw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

model.eval()
all_probs = []
with torch.no_grad():
    for gi, gm, y, sw in test_loader:
        gi, gm, y = gi.to(DEVICE), gm.to(DEVICE), y.to(DEVICE)
        all_probs.append(F.softmax(model(gi, gm), dim=1)[:, 1].cpu().numpy())
probs = np.concatenate(all_probs)
preds = (probs > 0.5).astype(int)

acc = accuracy_score(merged_labels, preds)
auc = roc_auc_score(merged_labels, probs)
cm = confusion_matrix(merged_labels, preds)
tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn) if (tp+fn)>0 else 0
spec = tn/(tn+fp) if (tn+fp)>0 else 0

print(f"\n=== DECONTAMINATED TRANSFER RESULTS ===")
print(f"clean->merged (decontaminated, n={len(merged_te_clean)}):")
print(f"  ACC={acc:.4f} AUC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f}")
print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"\nComparison:")
print(f"  Original (contaminated, n=838): ACC=0.6181 AUC=0.8060")
print(f"  Decontaminated (n={len(merged_te_clean)}): ACC={acc:.4f} AUC={auc:.4f}")

# Save
with open('/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis/decontaminated_transfer.json','w') as f:
    json.dump({
        'clean_train': len(clean_tr), 'merged_test_original': len(merged_te),
        'removed_overlap': len(merged_te)-len(keep),
        'decontaminated_test': len(keep),
        'acc': float(acc), 'auc': float(auc), 'sensitivity': float(sens), 'specificity': float(spec),
        'cm': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }, f, indent=2)
print("Saved decontaminated_transfer.json")
