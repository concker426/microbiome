#!/usr/bin/env python3
"""Regenerate FULL global SHAP ranking for Phase 3 — include ALL genera"""
import json,os,sys,pickle
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
MODEL_PATH='/hd/liujx/microbiome_llm_project/ProCyon_v2/backbone/final_model.pt'
OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
V=1226; E=768; DEVICE='cuda:1' if torch.cuda.is_available() else 'cpu'

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES=json.load(f)['genus_names']
print(f"Genus names: {len(GENUS_NAMES)}")

class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,E,padding_idx=0)
    def forward(self,ids,mask=None):
        x=self.emb(ids)
        mf=mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return x*mf, mf

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
        emb,mf=self.enc(ids,mask)
        pooled=(emb.sum(dim=1))/mf.sum(dim=1).clamp(min=1)
        return self.mlp(pooled)
    def get_per_genus_embeddings(self,ids,mask):
        emb,mf=self.enc(ids,mask)
        return emb, mf

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

@torch.no_grad()
def sample_importance(model,genus_ids,genus_mask,device):
    gi=torch.from_numpy(np.asarray(genus_ids).astype(np.int64)).long().unsqueeze(0).to(device)
    gm=torch.from_numpy(np.asarray(genus_mask)).bool().unsqueeze(0).to(device)
    emb,mf=model.get_per_genus_embeddings(gi,gm)
    valid=(gm[0]).cpu().numpy()
    n_valid=valid.sum()
    if n_valid<=1: return []

    pooled_full=(emb[0,valid].sum(dim=0))/n_valid
    logit_full=model.mlp(pooled_full.unsqueeze(0))
    prob_full=F.softmax(logit_full,dim=1)[0,1].item()

    emb_valid=emb[0,valid]
    sum_all=emb_valid.sum(dim=0)
    loo_means=[]
    for j in range(n_valid):
        loo_mean=(sum_all-emb_valid[j])/(n_valid-1)
        loo_means.append(loo_mean)
    loo_means=torch.stack(loo_means)
    logits_loo=model.mlp(loo_means)
    probs_loo=F.softmax(logits_loo,dim=1)[:,1]

    results=[]
    valid_indices=np.where(valid)[0]
    for j,pos in enumerate(valid_indices):
        gid=int(genus_ids[pos])
        if gid==0: continue
        gname=GENUS_NAMES[gid-1] if gid-1<len(GENUS_NAMES) else f'genus_{gid}'
        imp=prob_full-probs_loo[j].item()
        results.append({'genus_id':gid,'genus_name':gname,'importance':float(imp),
                        'prob_full':prob_full,'prob_without':float(probs_loo[j].item())})
    return results

if __name__=='__main__':
    print("Loading data...")
    train_data,test_data,ts,xs,tm,xm=load_data()
    model=Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True))
    model.eval()
    print(f"Model loaded. train={len(train_data)} test={len(test_data)}")

    # Compute ALL per-genus importance
    global_scores={}
    all_samples=[]
    total=len(train_data)+len(test_data)
    for i in range(len(train_data)):
        imp=sample_importance(model,ts[i],tm[i],DEVICE)
        if imp:
            all_samples.append({'sample_id':train_data[i]['sample_id'],'label':train_data[i]['label'],'importance':imp})
            for g in imp:
                gid=g['genus_id']; gname=g['genus_name']
                if gid not in global_scores: global_scores[gid]={'name':gname,'scores':[]}
                global_scores[gid]['scores'].append(g['importance'])
        if (i+1)%200==0: print(f"  Train: {i+1}/{len(train_data)}")

    for i in range(len(test_data)):
        imp=sample_importance(model,xs[i],xm[i],DEVICE)
        if imp:
            all_samples.append({'sample_id':test_data[i]['sample_id'],'label':test_data[i]['label'],'importance':imp})
            for g in imp:
                gid=g['genus_id']; gname=g['genus_name']
                if gid not in global_scores: global_scores[gid]={'name':gname,'scores':[]}
                global_scores[gid]['scores'].append(g['importance'])
        if (i+1)%100==0: print(f"  Test: {i+1}/{len(test_data)}")

    # Full global ranking
    global_ranking=[]
    for gid,info in global_scores.items():
        scores=info['scores']
        global_ranking.append({
            'genus_id':gid,'genus_name':info['name'],
            'mean_importance':float(np.mean(scores)),
            'std_importance':float(np.std(scores)),
            'n_samples':len(scores),
            'mean_abs_importance':float(np.mean(np.abs(scores)))
        })
    global_ranking.sort(key=lambda x:abs(x['mean_importance']),reverse=True)

    # Save FULL ranking
    with open(f'{OUT_DIR}/global_importance_full.csv','w') as f:
        f.write('rank,genus_id,genus_name,mean_importance,std_importance,n_samples,mean_abs_importance\n')
        for i,g in enumerate(global_ranking):
            f.write(f"{i+1},{g['genus_id']},{g['genus_name']},{g['mean_importance']:.6f},{g['std_importance']:.6f},{g['n_samples']},{g['mean_abs_importance']:.6f}\n")

    # Update pickle
    with open(f'{OUT_DIR}/shap_data_full.pkl','wb') as f:
        pickle.dump({'global_ranking':global_ranking,'all_samples':all_samples},f)

    print(f"\nTotal genera in ranking: {len(global_ranking)}")
    print(f"Saved: global_importance_full.csv, shap_data_full.pkl")
    print("DONE")
