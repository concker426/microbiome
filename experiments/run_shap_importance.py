#!/usr/bin/env python3
"""
Priority 3: SHAP-style leave-one-out feature importance
- For each sample: remove each genus, measure prediction change
- Output: global_top50.csv, patient_top20.csv, shap_data.pkl
"""
import json,os,sys,pickle,gc
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F

DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
OUT_DIR='/hd/liujx/microbiome_llm_project/experiments/results/final_backbone'
MODEL_PATH=f'{OUT_DIR}/models/final_model_s123.pt'
os.makedirs(OUT_DIR,exist_ok=True)

V=1226; E=768; DEVICE='cuda:1' if torch.cuda.is_available() else 'cpu'

# Load genus names
with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES=json.load(f)['genus_names']

class SimpleEmbEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,E,padding_idx=0)
    def forward(self,ids,mask=None):
        x=self.emb(ids)
        mf=mask.float().unsqueeze(-1) if mask is not None else torch.ones_like(x[...,:1])
        return x*mf, mf  # return masked embeddings and mask

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
        return emb, mf  # [B, SL, E], [B, SL, 1]

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
def sample_importance(model,genus_ids,genus_mask,sample_idx,device):
    """Leave-one-out importance for one sample. Returns list of (genus_id, genus_name, importance)."""
    gi=torch.from_numpy(np.asarray(genus_ids).astype(np.int64)).long().unsqueeze(0).to(device)
    gm=torch.from_numpy(np.asarray(genus_mask)).bool().unsqueeze(0).to(device)

    # Get per-genus embeddings
    emb,mf=model.get_per_genus_embeddings(gi,gm)  # [1, SL, E], [1, SL, 1]
    valid=(gm[0]).cpu().numpy()  # which positions have genera
    n_valid=valid.sum()
    if n_valid<=1: return []

    # Full prediction
    pooled_full=(emb[0,valid].sum(dim=0))/n_valid
    logit_full=model.mlp(pooled_full.unsqueeze(0))
    prob_full=F.softmax(logit_full,dim=1)[0,1].item()

    # Leave-one-out: batch all leave-one-out means
    emb_valid=emb[0,valid]  # [n_valid, E]
    sum_all=emb_valid.sum(dim=0)  # [E]
    loo_means=[]
    for j in range(n_valid):
        loo_mean=(sum_all-emb_valid[j])/(n_valid-1)
        loo_means.append(loo_mean)
    loo_means=torch.stack(loo_means)  # [n_valid, E]
    logits_loo=model.mlp(loo_means)  # [n_valid, 2]
    probs_loo=F.softmax(logits_loo,dim=1)[:,1]  # [n_valid]

    # Importance = change in disease probability when removing this genus
    results=[]
    valid_indices=np.where(valid)[0]
    for j,pos in enumerate(valid_indices):
        gid=int(genus_ids[pos])
        if gid==0: continue
        gname=GENUS_NAMES[gid-1] if gid-1<len(GENUS_NAMES) else f'genus_{gid}'
        imp=prob_full-probs_loo[j].item()  # positive = removing this genus decreases disease prob (genus associated with disease)
        results.append({'genus_id':gid,'genus_name':gname,'importance':float(imp),
                        'prob_full':prob_full,'prob_without':float(probs_loo[j].item())})
    return sorted(results,key=lambda x:abs(x['importance']),reverse=True)

if __name__=='__main__':
    print("="*50)
    print("Priority 3: SHAP Feature Importance")
    print("="*50)

    train_data,test_data,ts,xs,tm,xm=load_data()
    model=Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.eval()
    print(f"Model loaded. train={len(train_data)} test={len(test_data)}")

    # ── Compute importance for ALL samples ──
    all_importance=[]; global_scores={}
    for i in range(len(train_data)):
        imp=sample_importance(model,ts[i],tm[i],i,DEVICE)
        if imp:
            all_importance.append({'sample_id':train_data[i]['sample_id'],'label':train_data[i]['label'],'top20':imp[:20]})
            for g in imp:
                gid=g['genus_id']; gname=g['genus_name']
                if gid not in global_scores: global_scores[gid]={'name':gname,'scores':[]}
                global_scores[gid]['scores'].append(g['importance'])

    for i in range(len(test_data)):
        imp=sample_importance(model,xs[i],xm[i],i,DEVICE)
        if imp:
            all_importance.append({'sample_id':test_data[i]['sample_id'],'label':test_data[i]['label'],'top20':imp[:20]})
            for g in imp:
                gid=g['genus_id']; gname=g['genus_name']
                if gid not in global_scores: global_scores[gid]={'name':gname,'scores':[]}
                global_scores[gid]['scores'].append(g['importance'])

    # ── Global ranking ──
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

    # Save global top 50
    with open(f'{OUT_DIR}/global_importance.csv','w') as f:
        f.write('rank,genus_id,genus_name,mean_importance,std_importance,n_samples,mean_abs_importance\n')
        for i,g in enumerate(global_ranking[:50]):
            f.write(f"{i+1},{g['genus_id']},{g['genus_name']},{g['mean_importance']:.6f},{g['std_importance']:.6f},{g['n_samples']},{g['mean_abs_importance']:.6f}\n")

    # Save per-patient top 20
    with open(f'{OUT_DIR}/patient_top20.csv','w') as f:
        f.write('sample_id,label,rank,genus_id,genus_name,importance,prob_full,prob_without\n')
        for p in all_importance:
            for r,genus in enumerate(p['top20']):
                f.write(f"{p['sample_id']},{p['label']},{r+1},{genus['genus_id']},{genus['genus_name']},{genus['importance']:.6f},{genus['prob_full']:.4f},{genus['prob_without']:.4f}\n")

    # Save pickle
    with open(f'{OUT_DIR}/shap_data.pkl','wb') as f:
        pickle.dump({'all_importance':all_importance,'global_ranking':global_ranking},f)

    print(f"Top 10 global:")
    for i,g in enumerate(global_ranking[:10]):
        direction="↑Disease" if g['mean_importance']>0 else "↓Healthy"
        print(f"  {i+1}. {g['genus_name']:30s} imp={g['mean_importance']:+.4f} ±{g['std_importance']:.4f} [{direction}] (n={g['n_samples']})")
    print(f"\nSaved to {OUT_DIR}/")
    print("DONE")
