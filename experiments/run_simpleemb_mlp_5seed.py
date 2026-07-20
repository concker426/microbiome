#!/usr/bin/env python3
"""Job 1: SimpleEmb + Mean + MLP × 5 seeds stability. clean_2538. CPU/GPU."""
import json,os,sys,time,gc
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
RESULT_DIR='/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR,exist_ok=True)

V=1226; E=768; SL=86; NE=50; BS=32; LR=1e-3; WD=1e-4
SEEDS=[42,123,456,789,1024]
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
        self.fc1=nn.Linear(E,256)
        self.bn1=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(0.3)
        self.fc2=nn.Linear(256,2)
    def forward(self,x):
        x=self.fc1(x); x=F.relu(self.bn1(x)); x=self.drop(x)
        return self.fc2(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=SimpleEmbEnc(); self.mlp=MLP()
    def forward(self,ids,mask=None):
        return self.mlp(self.enc(ids,mask))

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

def make_label_map(train_data):
    labels=sorted(set(d['label'] for d in train_data))
    return {l:i for i,l in enumerate(labels)},labels

def build_ds(data,seqs,masks,label_map):
    class DS(Dataset):
        def __init__(self):
            self.seqs=seqs; self.masks=masks
            self.labels=[label_map[d['label']] for d in data]
            self.sw=[1.5 if d.get('label','Healthy')=='Disease' else 1.0 for d in data]
        def __len__(self): return len(self.labels)
        def __getitem__(self,i):
            return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                    torch.tensor(self.masks[i],dtype=torch.bool),
                    torch.tensor(self.labels[i],dtype=torch.long),
                    torch.tensor(self.sw[i],dtype=torch.float32))
    return DS()

def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
    y=torch.stack([x[2] for x in batch]); sw=torch.stack([x[3] for x in batch])
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y,sw

if __name__=='__main__':
    print("="*60)
    print("Job 1: SimpleEmb + Mean + MLP x 5 seeds")
    print("="*60)
    train_data,test_data,ts,xs,tm,xm=load_data()
    label_map,label_names=make_label_map(train_data)
    print(f"train={len(train_data)} test={len(test_data)} labels={label_names} device={DEVICE}")
    train_ds=build_ds(train_data,ts,tm,label_map)
    test_ds=build_ds(test_data,xs,xm,label_map)
    train_loader=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=BS,shuffle=False,collate_fn=collate)

    results=[]
    for seed in SEEDS:
        t0=time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        model=Model().to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NE)

        # Train
        for ep in range(NE):
            model.train(); ep_loss=0; n=0
            for gi,gm,y,sw in train_loader:
                gi,gm,y,sw=gi.to(DEVICE),gm.to(DEVICE),y.to(DEVICE),sw.to(DEVICE)
                logits=model(gi,gm)
                loss=F.cross_entropy(logits,y,reduction='none')
                loss=(loss*sw).sum()/sw.sum()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss+=loss.item(); n+=1
            sched.step()

        # Eval
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for gi,gm,y,sw in test_loader:
                gi,gm,y=gi.to(DEVICE),gm.to(DEVICE),y.to(DEVICE)
                logits=model(gi,gm); pred=torch.argmax(logits,dim=1)
                correct+=(pred==y).sum().item(); total+=y.size(0)
        acc=correct/total
        print(f"  Seed {seed}: ACC={acc:.4f} time={time.time()-t0:.0f}s")
        results.append({'seed':seed,'accuracy':acc})
        del model; gc.collect(); torch.cuda.empty_cache()

    accs=[r['accuracy'] for r in results]
    mean_acc=np.mean(accs); std_acc=np.std(accs)
    print(f"\n{'='*40}")
    print(f"  MEAN: {mean_acc:.4f} +/- {std_acc:.4f}")
    for r in results: print(f"    Seed {r['seed']}: {r['accuracy']:.4f}")
    print(f"{'='*40}")

    output={'experiment':'simpleemb_mlp_5seed','dataset':'clean_2538',
            'n_train':len(train_data),'n_test':len(test_data),
            'seeds':SEEDS,'results':results,
            'mean':float(mean_acc),'std':float(std_acc)}
    with open(f'{RESULT_DIR}/simpleemb_mlp_5seed.json','w') as f:
        json.dump(output,f,indent=2)
    print("Saved to simpleemb_mlp_5seed.json")

    if std_acc<0.03 and mean_acc>0.89:
        print("STABLE. SimpleEmb+MLP baseline CONFIRMED.")
    else:
        print(f"Check stability (std={std_acc:.4f}).")
    print("DONE")
