#!/usr/bin/env python3
"""
Priority 1+2: SimpleEmb + Mean + MLP final backbone
- 5-fold Stratified CV × 5 seeds
- Save: model, confusion matrix, ROC, PR, per-sample probs, embeddings, patient IDs
- Export 768-d embeddings for ALL samples
"""
import json,os,sys,time,gc,pickle
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix,roc_auc_score,roc_curve,
    average_precision_score,precision_recall_curve,accuracy_score)

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
OUT_DIR='/hd/liujx/microbiome_llm_project/experiments/results/final_backbone'
os.makedirs(OUT_DIR,exist_ok=True)
os.makedirs(f'{OUT_DIR}/models',exist_ok=True)
os.makedirs(f'{OUT_DIR}/embeddings',exist_ok=True)

V=1226; E=768; SL=86; NE=50; BS=32; LR=1e-3; WD=1e-4
SEEDS=[42,123,456,789,1024]; N_FOLDS=5
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

def build_ds(data,seqs,masks,label_map):
    class DS(Dataset):
        def __init__(self):
            self.seqs=seqs; self.masks=masks
            self.labels=np.array([label_map[d['label']] for d in data])
            self.ids=[d['sample_id'] for d in data]
            self.sw=np.array([1.5 if d.get('label','Healthy')=='Disease' else 1.0 for d in data])
        def __len__(self): return len(self.labels)
        def __getitem__(self,i):
            return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                    torch.tensor(self.masks[i],dtype=torch.bool),
                    torch.tensor(self.labels[i],dtype=torch.long),
                    torch.tensor(self.sw[i],dtype=torch.float32), i)
    return DS()

def collate(batch):
    gi=[x[0] for x in batch]; gm=[x[1] for x in batch]
    y=torch.stack([x[2] for x in batch]); sw=torch.stack([x[3] for x in batch])
    idx=torch.tensor([x[4] for x in batch],dtype=torch.long)
    mgl=max(len(g) for g in gi); pg,pm=[],[]
    for i in range(len(gi)):
        g=gi[i]; m=gm[i]; p=mgl-len(g)
        pg.append(torch.cat([g,torch.zeros(p,dtype=torch.long)]) if p>0 else g)
        pm.append(torch.cat([m,torch.zeros(p,dtype=torch.bool)]) if p>0 else m)
    return torch.stack(pg),torch.stack(pm),y,sw,idx

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); all_preds=[]; all_probs=[]; all_labels=[]; all_idx=[]
    for gi,gm,y,sw,idx in loader:
        gi,gm,y=gi.to(DEVICE),gm.to(DEVICE),y.to(DEVICE)
        logits=model(gi,gm); prob=F.softmax(logits,dim=1)
        all_preds.append(torch.argmax(logits,dim=1).cpu().numpy())
        all_probs.append(prob[:,1].cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_idx.append(idx.numpy())
    return (np.concatenate(all_preds),np.concatenate(all_probs),
            np.concatenate(all_labels),np.concatenate(all_idx))

@torch.no_grad()
def get_embeddings(model,loader):
    model.eval(); all_emb=[]; all_idx=[]
    for gi,gm,y,sw,idx in loader:
        gi,gm=gi.to(DEVICE),gm.to(DEVICE)
        emb=model.encode(gi,gm)
        all_emb.append(emb.cpu().numpy()); all_idx.append(idx.numpy())
    return np.concatenate(all_emb),np.concatenate(all_idx)

def train_epochs(model,loader,opt,sched,epochs):
    for ep in range(epochs):
        model.train(); ep_loss=0; n=0
        for gi,gm,y,sw,idx in loader:
            gi,gm,y,sw=gi.to(DEVICE),gm.to(DEVICE),y.to(DEVICE),sw.to(DEVICE)
            logits=model(gi,gm)
            loss=F.cross_entropy(logits,y,reduction='none')
            loss=(loss*sw).sum()/sw.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss+=loss.item(); n+=1
        sched.step()

if __name__=='__main__':
    print("="*60)
    print("Priority 1+2: Final Backbone — SimpleEmb + Mean + MLP")
    print("="*60)

    train_data,test_data,ts,xs,tm,xm=load_data()
    label_map={'Healthy':0,'Disease':1}; label_names=['Healthy','Disease']
    print(f"train={len(train_data)} test={len(test_data)}")

    train_ds=build_ds(train_data,ts,tm,label_map)
    test_ds=build_ds(test_data,xs,xm,label_map)
    train_labels=np.array([label_map[d['label']] for d in train_data])

    all_results={}; all_fold_preds=[]; all_fold_embs=[]

    # ── 5-Fold CV × 5 Seeds ──
    for seed in SEEDS:
        print(f"\n{'='*40}\nSeed={seed}\n{'='*40}")
        torch.manual_seed(seed); np.random.seed(seed)
        skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
        fold_metrics=[]
        for fold,(train_idx,val_idx) in enumerate(skf.split(np.zeros(len(train_labels)),train_labels)):
            t0=time.time()
            model=Model().to(DEVICE)
            opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NE)

            # Build fold datasets
            fold_train=torch.utils.data.Subset(train_ds,train_idx)
            fold_val=torch.utils.data.Subset(train_ds,val_idx)
            train_loader=DataLoader(fold_train,batch_size=BS,shuffle=True,collate_fn=collate)
            val_loader=DataLoader(fold_val,batch_size=BS,shuffle=False,collate_fn=collate)

            train_epochs(model,train_loader,opt,sched,NE)
            preds,probs,labels,idx=evaluate(model,val_loader)

            acc=accuracy_score(labels,preds)
            try: auc=roc_auc_score(labels,probs)
            except: auc=0.0
            try: ap=average_precision_score(labels,probs)
            except: ap=0.0
            cm=confusion_matrix(labels,preds).tolist()

            fold_metrics.append({'fold':fold,'acc':float(acc),'auc':float(auc),'ap':float(ap),'cm':cm})
            print(f"  Fold {fold}: ACC={acc:.4f} AUC={auc:.4f} AP={ap:.4f} time={time.time()-t0:.0f}s")

            # Save best model per fold
            torch.save(model.state_dict(),f'{OUT_DIR}/models/model_s{seed}_f{fold}.pt')
            del model; gc.collect()

        accs=[m['acc'] for m in fold_metrics]
        print(f"  Seed {seed} CV: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        all_results[f'seed_{seed}']={'folds':fold_metrics,'mean':float(np.mean(accs)),'std':float(np.std(accs))}

    # ── Final model: train on ALL train data, eval on test ──
    print(f"\n{'='*40}\nFinal Model (all train → test)")
    final_results={}
    for seed in SEEDS:
        t0=time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        model=Model().to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NE)

        all_train_loader=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=collate)
        test_loader=DataLoader(test_ds,batch_size=BS,shuffle=False,collate_fn=collate)

        train_epochs(model,all_train_loader,opt,sched,NE)
        preds,probs,labels,idx=evaluate(model,test_loader)

        acc=accuracy_score(labels,preds)
        try: auc=roc_auc_score(labels,probs)
        except: auc=0.0
        try: ap=average_precision_score(labels,probs)
        except: ap=0.0
        cm=confusion_matrix(labels,preds).tolist()
        fpr,tpr,_=roc_curve(labels,probs)
        prec,recall,_=precision_recall_curve(labels,probs)

        # Per-sample predictions
        sample_preds=[]
        for i in range(len(preds)):
            sample_preds.append({
                'sample_id': test_ds.ids[int(idx[i])],
                'ground_truth': label_names[int(labels[i])],
                'predicted': label_names[int(preds[i])],
                'prob_disease': float(probs[i])
            })

        final_results[f'seed_{seed}']={'acc':float(acc),'auc':float(auc),'ap':float(ap),'cm':cm,
            'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},
            'pr':{'precision':prec.tolist(),'recall':recall.tolist()},
            'predictions':sample_preds}

        torch.save(model.state_dict(),f'{OUT_DIR}/models/final_model_s{seed}.pt')
        print(f"  Seed {seed}: ACC={acc:.4f} AUC={auc:.4f} AP={ap:.4f} time={time.time()-t0:.0f}s")
        del model; gc.collect()

    # ── Export embeddings for ALL samples (best seed = 123) ──
    print(f"\n{'='*40}\nExporting Embeddings")
    best_seed=123
    torch.manual_seed(best_seed); np.random.seed(best_seed)
    model=Model().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NE)
    all_train_loader=DataLoader(train_ds,batch_size=BS,shuffle=True,collate_fn=collate)
    train_epochs(model,all_train_loader,opt,sched,NE)

    # Export train embeddings
    train_emb_loader=DataLoader(train_ds,batch_size=BS,shuffle=False,collate_fn=collate)
    emb_train,idx_train=get_embeddings(model,train_emb_loader)
    # Export test embeddings
    test_emb_loader=DataLoader(test_ds,batch_size=BS,shuffle=False,collate_fn=collate)
    emb_test,idx_test=get_embeddings(model,test_emb_loader)

    # Build CSV
    rows=[['split','sample_id','label']+[f'dim_{i}' for i in range(E)]]
    for i in range(len(emb_train)):
        sid=train_ds.ids[int(idx_train[i])]; lab=label_names[int(train_labels[int(idx_train[i])])]
        rows.append(['train',sid,lab]+[f'{v:.6f}' for v in emb_train[i]])
    for i in range(len(emb_test)):
        sid=test_ds.ids[int(idx_test[i])]; lab=label_names[int(test_ds.labels[int(idx_test[i])])]
        rows.append(['test',sid,lab]+[f'{v:.6f}' for v in emb_test[i]])

    with open(f'{OUT_DIR}/embeddings/representation.csv','w') as f:
        for r in rows: f.write(','.join(r)+'\n')

    np.save(f'{OUT_DIR}/embeddings/train_embeddings.npy',emb_train)
    np.save(f'{OUT_DIR}/embeddings/test_embeddings.npy',emb_test)
    np.save(f'{OUT_DIR}/embeddings/train_ids.npy',np.array([train_ds.ids[int(i)] for i in idx_train]))
    np.save(f'{OUT_DIR}/embeddings/test_ids.npy',np.array([test_ds.ids[int(i)] for i in idx_test]))
    np.save(f'{OUT_DIR}/embeddings/train_labels.npy',np.array([train_labels[int(i)] for i in idx_train]))
    np.save(f'{OUT_DIR}/embeddings/test_labels.npy',np.array([test_ds.labels[int(i)] for i in idx_test]))
    print(f"  Train embeddings: {emb_train.shape}")
    print(f"  Test embeddings:  {emb_test.shape}")

    # ── Save all metadata ──
    output={
        'experiment':'final_backbone','dataset':'clean_2538',
        'n_train':len(train_data),'n_test':len(test_data),
        'cv_results':all_results,
        'final_results':final_results,
        'label_map':label_map,'label_names':label_names
    }
    with open(f'{OUT_DIR}/final_backbone_results.json','w') as f:
        json.dump(output,f,indent=2,default=str)
    print(f"\nAll saved to {OUT_DIR}")
    print("DONE")
