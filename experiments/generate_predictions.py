#!/usr/bin/env python3
"""Generate per-sample predictions (ensemble across 5 seeds) for both train+test"""
import json,os,sys
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
MODEL_DIR='/hd/liujx/microbiome_llm_project/experiments/results/final_backbone/models'
OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/backbone'
os.makedirs(OUT_DIR,exist_ok=True)

V=1226; E=768; SL=86; BS=32; SEEDS=[42,123,456,789,1024]
DEVICE='cuda:1' if torch.cuda.is_available() else 'cpu'

class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,E,padding_idx=0)
    def forward(self,ids,mask=None):
        x=self.emb(ids)
        mf=mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return (x*mf).sum(dim=1)/mf.sum(dim=1).clamp(min=1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(E,256); self.bn1=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(0.3); self.fc2=nn.Linear(256,2)
    def forward(self,x):
        x=self.fc1(x); x=F.relu(self.bn1(x)); x=self.drop(x)
        return self.fc2(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=SimpleEmbEnc(); self.mlp=MLP()
    def forward(self,ids,mask=None):
        return self.mlp(self.enc(ids,mask))
    def encode(self,ids,mask=None):
        return self.enc(ids,mask)

def load_data():
    train_data,test_data=[],[]
    with open(f'{DATA_DIR}/train_nl.jsonl') as f:
        for l in f: train_data.append(json.loads(l))
    with open(f'{DATA_DIR}/test_nl.jsonl') as f:
        for l in f: test_data.append(json.loads(l))
    ts=np.load(f'{DATA_DIR}/train_genus_sequences.npy')
    xs=np.load(f'{DATA_DIR}/test_genus_sequences.npy')
    tm=np.load(f'{DATA_DIR}/train_genus_masks.npy')
    xm=np.load(f'{DATA_DIR}/test_genus_masks.npy')
    return train_data,test_data,ts,xs,tm,xm

def build_ds(data,seqs,masks):
    class DS(Dataset):
        def __init__(self):
            self.seqs=seqs; self.masks=masks
            self.labels=np.array([0 if d['label']=='Healthy' else 1 for d in data])
            self.ids=[d['sample_id'] for d in data]
        def __len__(self): return len(self.labels)
        def __getitem__(self,i):
            return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                    torch.tensor(self.masks[i],dtype=torch.bool),
                    torch.tensor(self.labels[i],dtype=torch.long), i)
    return DS()

def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
    y=torch.stack([x[2] for x in batch])
    idx=torch.tensor([x[3] for x in batch],dtype=torch.long)
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y,idx

@torch.no_grad()
def get_probs(model,loader):
    model.eval(); all_probs=[]; all_labels=[]; all_idx=[]
    for gi,gm,y,idx in loader:
        gi,gm=gi.to(DEVICE),gm.to(DEVICE)
        logits=model(gi,gm); prob=F.softmax(logits,dim=1)
        all_probs.append(prob[:,1].cpu().numpy())
        all_labels.append(y.numpy())
        all_idx.append(idx.numpy())
    return np.concatenate(all_probs),np.concatenate(all_labels),np.concatenate(all_idx)

if __name__=='__main__':
    print("Loading data...")
    train_data,test_data,ts,xs,tm,xm=load_data()
    train_ds=build_ds(train_data,ts,tm)
    test_ds=build_ds(test_data,xs,xm)
    label_names=['Healthy','Disease']

    train_loader=DataLoader(train_ds,batch_size=BS,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=BS,shuffle=False,collate_fn=collate)

    # Ensemble probs across 5 seeds
    train_probs_ens=np.zeros(len(train_ds))
    test_probs_ens=np.zeros(len(test_ds))

    for seed in SEEDS:
        print(f"Loading seed={seed}...")
        model=Model().to(DEVICE)
        sd=torch.load(f'{MODEL_DIR}/final_model_s{seed}.pt',map_location=DEVICE,weights_only=True)
        model.load_state_dict(sd)

        probs_t,labels_t,idx_t=get_probs(model,train_loader)
        probs_s,labels_s,idx_s=get_probs(model,test_loader)

        train_probs_ens[idx_t]+=probs_t
        test_probs_ens[idx_s]+=probs_s
        del model

    train_probs_ens/=len(SEEDS)
    test_probs_ens/=len(SEEDS)
    train_preds=(train_probs_ens>0.5).astype(int)
    test_preds=(test_probs_ens>0.5).astype(int)

    # Build combined predictions CSV
    rows=[['sample_id','split','ground_truth','prob_disease','predicted','correct']]
    for i in range(len(train_ds)):
        rows.append([
            train_ds.ids[i],'train',
            label_names[int(train_ds.labels[i])],
            f'{train_probs_ens[i]:.6f}',
            label_names[int(train_preds[i])],
            str(train_preds[i]==train_ds.labels[i])
        ])
    for i in range(len(test_ds)):
        rows.append([
            test_ds.ids[i],'test',
            label_names[int(test_ds.labels[i])],
            f'{test_probs_ens[i]:.6f}',
            label_names[int(test_preds[i])],
            str(test_preds[i]==test_ds.labels[i])
        ])

    with open(f'{OUT_DIR}/predictions.csv','w') as f:
        for r in rows: f.write(','.join(r)+'\n')

    # Accuracy summary
    train_acc=(train_preds==train_ds.labels).mean()
    test_acc=(test_preds==test_ds.labels).mean()
    print(f"\nEnsemble results (5 seeds):")
    print(f"  Train ACC: {train_acc:.4f} ({train_preds.sum()} disease / {(1-train_preds).sum()} healthy)")
    print(f"  Test  ACC: {test_acc:.4f} ({test_preds.sum()} disease / {(1-test_preds).sum()} healthy)")
    print(f"\nSaved: {OUT_DIR}/predictions.csv ({len(rows)-1} samples)")
