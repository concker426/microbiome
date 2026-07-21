#!/usr/bin/env python3
"""Phase 3.6: Biomarker Robustness — CV SHAP stability, permutation control, abundance analysis"""
import sys,os,csv,json,pickle
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
DATA_DIR='/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
MODEL_DIR='/hd/liujx/microbiome_llm_project/experiments/results/final_backbone/models'
os.makedirs(OUT_DIR,exist_ok=True)

V=1226; E=768; DEVICE='cuda:1' if torch.cuda.is_available() else 'cpu'
SEED_REF=42

with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES=json.load(f)['genus_names']

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
        return self.mlp((emb.sum(dim=1))/mf.sum(dim=1).clamp(min=1))
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
    emb_valid=emb[0,valid]; sum_all=emb_valid.sum(dim=0)
    loo_means=[(sum_all-emb_valid[j])/(n_valid-1) for j in range(n_valid)]
    loo_means=torch.stack(loo_means)
    probs_loo=F.softmax(model.mlp(loo_means),dim=1)[:,1]
    results=[]
    valid_indices=np.where(valid)[0]
    for j,pos in enumerate(valid_indices):
        gid=int(genus_ids[pos])
        if gid==0: continue
        gname=GENUS_NAMES[gid-1] if gid-1<len(GENUS_NAMES) else f'genus_{gid}'
        imp=prob_full-probs_loo[j].item()
        results.append({'genus_id':gid,'genus_name':gname,'importance':imp,
                        'prob_full':prob_full,'prob_without':float(probs_loo[j].item())})
    return results

def aggregate_global(shap_results):
    scores={}
    for r in shap_results:
        for g in r['importance']:
            gid=g['genus_id']; gname=g['genus_name']
            if gid not in scores: scores[gid]={'name':gname,'scores':[]}
            scores[gid]['scores'].append(g['importance'])
    ranking=[]
    for gid,info in scores.items():
        sc=info['scores']
        ranking.append({'genus_id':gid,'genus_name':info['name'],
            'mean_importance':float(np.mean(sc)),'n_samples':len(sc),
            'mean_abs_importance':float(np.mean(np.abs(sc)))})
    ranking.sort(key=lambda x:abs(x['mean_importance']),reverse=True)
    return ranking

def jaccard(set_a,set_b):
    if not set_a or not set_b: return 0.0
    return len(set_a&set_b)/len(set_a|set_b)

print("="*60)
print("Phase 3.6: Biomarker Robustness Analysis")
print("="*60)

train_data,test_data,ts,xs,tm,xm=load_data()
train_labels=np.array([0 if d['label']=='Healthy' else 1 for d in train_data])
print(f"train={len(train_data)} test={len(test_data)}")

# ═══════════════════════════════════════════════
# 3.6.1 CV SHAP Stability
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("3.6.1: Cross-Validation SHAP Stability")
print(f"{'='*60}")

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED_REF)
fold_topk={k:[] for k in [10,20,50]}

for fold,(train_idx,val_idx) in enumerate(skf.split(np.zeros(len(train_labels)),train_labels)):
    print(f"  Fold {fold}: val n={len(val_idx)}")

    # Load fold model
    model=Model().to(DEVICE)
    sd=torch.load(f'{MODEL_DIR}/model_s{SEED_REF}_f{fold}.pt',map_location=DEVICE,weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    # Compute SHAP on validation samples
    fold_results=[]
    for idx in val_idx:
        imp=sample_importance(model,ts[idx],tm[idx],DEVICE)
        if imp:
            fold_results.append({
                'sample_id':train_data[idx]['sample_id'],
                'label':train_data[idx]['label'],
                'importance':imp
            })

    ranking=aggregate_global(fold_results)
    for k in [10,20,50]:
        topk_names=set(r['genus_name'] for r in ranking[:k])
        fold_topk[k].append(topk_names)
        print(f"    Top-{k}: {len(topk_names)} unique genera")

    # Quick accuracy check
    preds=[]; lbls=[]
    for r in fold_results:
        probs=[g['prob_full'] for g in r['importance'] if 'prob_full' in dir(r)]
        if r['importance']:
            prob_disease=r['importance'][0]['prob_full']
            preds.append(1 if prob_disease>0.5 else 0)
            lbls.append(0 if r['label']=='Healthy' else 1)
    if preds:
        acc=accuracy_score(lbls,preds)
        print(f"    Val ACC (from saved probs): {acc:.4f}")

    del model

# Compute pairwise Jaccard
jaccard_results={}
for k in [10,20,50]:
    sets=fold_topk[k]
    pairs=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            pairs.append(jaccard(sets[i],sets[j]))
    mean_j=np.mean(pairs); std_j=np.std(pairs)
    jaccard_results[k]={'mean':float(mean_j),'std':float(std_j),'pairs':[float(p) for p in pairs]}
    print(f"\n  Top-{k}: Mean Jaccard = {mean_j:.4f} ± {std_j:.4f}")
    print(f"    Pairwise: {[f'{p:.3f}' for p in pairs]}")

# Find consistent genera (appear in all 5 folds)
for k in [20,50]:
    consistent=fold_topk[k][0].copy()
    for s in fold_topk[k][1:]:
        consistent&=s
    print(f"  Genera in ALL 5 folds Top-{k}: {len(consistent)} — {sorted(consistent)[:15]}")

# ═══════════════════════════════════════════════
# 3.6.2 Permutation Importance Control
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("3.6.2: Permutation Importance Control")
print(f"{'='*60}")

# Shuffle labels
shuffled_labels=train_labels.copy()
np.random.RandomState(42).shuffle(shuffled_labels)
shuffled_acc=(shuffled_labels==train_labels).mean()
print(f"  Label shuffle: {shuffled_acc:.4f} match with original (should be ~0.5)")

# Train a model with shuffled labels (quick: 30 epochs)
print("  Training model with shuffled labels...")
torch.manual_seed(SEED_REF); np.random.seed(SEED_REF)
perm_model=Model().to(DEVICE)
opt=torch.optim.AdamW(perm_model.parameters(),lr=1e-3,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)

# Build dataset manually for shuffled labels
class PermDS:
    def __init__(self,seqs,masks,labels,ids):
        self.seqs=seqs; self.masks=masks; self.labels=labels; self.ids=ids
        self.sw=np.array([1.5 if l==1 else 1.0 for l in labels])
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        return (torch.tensor(self.seqs[i].astype(np.int64),dtype=torch.long),
                torch.tensor(self.masks[i],dtype=torch.bool),
                torch.tensor(self.labels[i],dtype=torch.long),
                torch.tensor(self.sw[i],dtype=torch.float32), i)

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

perm_ds=PermDS(ts,tm,shuffled_labels,[d['sample_id'] for d in train_data])
from torch.utils.data import DataLoader
perm_loader=DataLoader(perm_ds,batch_size=32,shuffle=True,collate_fn=collate)
for ep in range(30):
    perm_model.train()
    for gi,gm,y,sw,idx in perm_loader:
        gi,gm,y,sw=gi.to(DEVICE),gm.to(DEVICE),y.to(DEVICE),sw.to(DEVICE)
        logits=perm_model(gi,gm)
        loss=F.cross_entropy(logits,y,reduction='none')
        loss=(loss*sw).sum()/sw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
perm_model.eval()

# Check: shuffled model accuracy should be ~50% on train
correct=0; total=0
for gi,gm,y,sw,idx in perm_loader:
    gi,gm=gi.to(DEVICE),gm.to(DEVICE)
    logits=perm_model(gi,gm)
    preds=torch.argmax(logits,dim=1).cpu()
    correct+=(preds==y).sum().item(); total+=len(y)
print(f"  Shuffled model train ACC: {correct/total:.4f} (should be near 0.5)")

# Compute SHAP on a subset for permutation model
print("  Computing SHAP for permuted model...")
perm_shap=[]
for i in range(min(100,len(train_data))):
    imp=sample_importance(perm_model,ts[i],tm[i],DEVICE)
    if imp: perm_shap.append({'sample_id':train_data[i]['sample_id'],'importance':imp})
perm_ranking=aggregate_global(perm_shap)
del perm_model

# Compare real vs permuted rankings
real_ranking_names=[r['genus_name'] for r in aggregate_global([])]
# Load existing full ranking
with open(f'{OUT_DIR}/global_importance_full.csv') as f:
    real_ranking=[r for r in csv.DictReader(f)]
real_top20=set(r['genus_name'] for r in real_ranking[:20])
perm_top20=set(r['genus_name'] for r in perm_ranking[:20])

print(f"  Real top-20 ∩ Permuted top-20: {len(real_top20 & perm_top20)}")
print(f"  Jaccard (real vs permuted top-20): {jaccard(real_top20,perm_top20):.4f}")

# Key comparison: permuted model's mean |SHAP| distribution
real_abs_imps=[abs(float(r['mean_importance'])) for r in real_ranking]
perm_abs_imps=[abs(r['mean_importance']) for r in perm_ranking]
print(f"  Real  |SHAP|: mean={np.mean(real_abs_imps):.6f} max={np.max(real_abs_imps):.6f}")
print(f"  Perm  |SHAP|: mean={np.mean(perm_abs_imps):.6f} max={np.max(perm_abs_imps):.6f}")
print(f"  Ratio (real/perm mean): {np.mean(real_abs_imps)/np.mean(perm_abs_imps):.2f}x")

# ═══════════════════════════════════════════════
# 3.6.3 Abundance vs Importance Analysis
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("3.6.3: Abundance vs Importance")
print(f"{'='*60}")

# Compute per-genus statistics from raw data
genus_prevalence={}; genus_mean_abundance={}
for i in range(len(train_data)):
    seq=ts[i]; mask=tm[i].astype(bool)
    # Count which genera appear and compute mean abundance rank (lower = more abundant)
    for pos in range(len(seq)):
        if mask[pos] and seq[pos]>0:
            gid=int(seq[pos])
            if gid not in genus_prevalence: genus_prevalence[gid]=0
            genus_prevalence[gid]+=1
            if gid not in genus_mean_abundance: genus_mean_abundance[gid]=[]
            genus_mean_abundance[gid].append(pos)  # position in ranked list (0=most abundant)

n_train=len(train_data)

# Merge with SHAP data
with open(f'{OUT_DIR}/global_importance_full.csv') as f:
    shap_full=list(csv.DictReader(f))

abundance_data=[]
for r in shap_full:
    gid=int(r['genus_id']); gname=r['genus_name']
    imp=abs(float(r['mean_importance']))
    prev=genus_prevalence.get(gid,0)/n_train
    mean_pos=np.mean(genus_mean_abundance.get(gid,[86])) if gid in genus_mean_abundance else 86
    abundance_data.append({
        'genus_name':gname,'genus_id':gid,
        'abs_importance':imp,'prevalence':prev,
        'mean_abundance_rank':mean_pos,'n_samples':int(r['n_samples'])
    })

# Key finding: are top SHAP genera high-abundance or low-abundance?
top20_shap=abundance_data[:20]
top20_prev=[g['prevalence'] for g in top20_shap]
top20_rank=[g['mean_abundance_rank'] for g in top20_shap]
print(f"  Top-20 SHAP mean prevalence: {np.mean(top20_prev):.4f}")
print(f"  Top-20 SHAP mean abundance rank: {np.mean(top20_rank):.1f} (lower = more abundant)")

# All genera prevalence
all_prev=[g['prevalence'] for g in abundance_data]
all_imp=[g['abs_importance'] for g in abundance_data]
print(f"  All genera mean prevalence: {np.mean(all_prev):.4f}")
print(f"  Correlation (prevalence vs importance): r={np.corrcoef(all_prev,all_imp)[0,1]:.4f}")

# Find genera with HIGH importance but LOW prevalence
novel_candidates=[]
for g in abundance_data:
    if g['prevalence']<0.1 and g['abs_importance']>np.percentile(all_imp,90):
        novel_candidates.append(g)
novel_candidates.sort(key=lambda x: x['abs_importance'],reverse=True)
print(f"\n  Novel candidates (prevalence<10%, importance>P90):")
for g in novel_candidates[:10]:
    print(f"    {g['genus_name']:30s} |SHAP|={g['abs_importance']:.6f} prev={g['prevalence']:.3f} rank={g['mean_abundance_rank']:.0f}")

# ═══════════════════════════════════════════════
# Phase 3.5+: Cluster 1 Validation
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("3.5+: Cluster 1 Outlier Validation")
print(f"{'='*60}")

with open(f'{OUT_DIR}/heterogeneity_results.json') as f:
    het=json.load(f)

emb=np.load(f'{OUT_DIR}/../backbone/embeddings.npy')
labels_all=np.load(f'{OUT_DIR}/../backbone/labels.npy')
ids_all=np.load(f'{OUT_DIR}/../backbone/sample_ids.npy')

# Map heterogeneity results
cluster_map=het['cluster_assignments']
emb_train_d=emb[(labels_all==1)]; ids_train_d=ids_all[(labels_all==1)]
c0_sids=set(sid for sid,c in cluster_map.items() if c==0)
c1_sids=set(sid for sid,c in cluster_map.items() if c==1)

# Compute per-sample diversity: number of unique genera (species richness proxy)
c0_richness=[]; c1_richness=[]
for i in range(len(train_data)):
    sid=train_data[i]['sample_id']
    seq=ts[i]; mask=tm[i].astype(bool)
    n_genera=mask.sum()  # number of valid genus positions
    if sid in c0_sids: c0_richness.append(n_genera)
    if sid in c1_sids: c1_richness.append(n_genera)

print(f"  Cluster 0 (n={len(c0_richness)}): mean genera = {np.mean(c0_richness):.1f} ± {np.std(c0_richness):.1f}")
print(f"  Cluster 1 (n={len(c1_richness)}): mean genera = {np.mean(c1_richness):.1f} ± {np.std(c1_richness):.1f}")

# Distance to healthy centroid
emb_healthy=emb[(labels_all==0)]
centroid_h=emb_healthy.mean(0)
centroid_h_n=centroid_h/np.linalg.norm(centroid_h)

c0_dists=[]; c1_dists=[]
for i in range(len(train_data)):
    sid=train_data[i]['sample_id']
    if labels_all[i]==0: continue  # skip healthy
    e=emb[i]; e_n=e/np.linalg.norm(e)
    dist=1-(e_n@centroid_h_n)
    if sid in c0_sids: c0_dists.append(dist)
    if sid in c1_sids: c1_dists.append(dist)

print(f"  Cluster 0: distance to Healthy = {np.mean(c0_dists):.4f} ± {np.std(c0_dists):.4f}")
print(f"  Cluster 1: distance to Healthy = {np.mean(c1_dists):.4f} ± {np.std(c1_dists):.4f}")

# Prediction entropy (uncertainty proxy)
c0_entropy=[]; c1_entropy=[]
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data=list(csv.DictReader(f))
for r in pred_data:
    if r['split']!='train': continue
    sid=r['sample_id']
    p=float(r['prob_disease'])
    # Binary entropy
    p_clip=np.clip(p,1e-6,1-1e-6)
    ent=-(p_clip*np.log(p_clip)+(1-p_clip)*np.log(1-p_clip))
    if sid in c0_sids: c0_entropy.append(ent)
    if sid in c1_sids: c1_entropy.append(ent)

print(f"  Cluster 0: mean entropy = {np.mean(c0_entropy):.4f} (higher = less certain)")
print(f"  Cluster 1: mean entropy = {np.mean(c1_entropy):.4f}")

# ═══════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════
print(f"\nGenerating figures...")
fig,axes=plt.subplots(2,3,figsize=(18,12))

# Panel 1: Jaccard bar chart
ax=axes[0,0]
ks=[10,20,50]
means=[jaccard_results[k]['mean'] for k in ks]
stds=[jaccard_results[k]['std'] for k in ks]
ax.bar(range(len(ks)),means,yerr=stds,color=['#FF9800','#4CAF50','#2196F3'],capsize=5,edgecolor='none')
ax.set_xticks(range(len(ks))); ax.set_xticklabels([f'Top-{k}' for k in ks])
ax.set_ylabel('Mean Jaccard Similarity'); ax.set_title('CV SHAP Stability (5 folds)')
ax.set_ylim(0,1); ax.grid(True,alpha=0.3,axis='y')
for i,(m,s) in enumerate(zip(means,stds)):
    ax.text(i,m+s+0.02,f'{m:.3f}',ha='center',fontsize=10)

# Panel 2: Real vs Permuted SHAP distribution
ax=axes[0,1]
bins=np.linspace(0,max(max(real_abs_imps),max(perm_abs_imps))*0.5,40)
ax.hist(real_abs_imps,bins=bins,alpha=0.6,label=f'Real (mean={np.mean(real_abs_imps):.4f})',color='#4CAF50')
ax.hist(perm_abs_imps,bins=bins,alpha=0.6,label=f'Permuted (mean={np.mean(perm_abs_imps):.4f})',color='#F44336')
ax.set_xlabel('|SHAP Importance|'); ax.set_ylabel('Frequency')
ax.set_title('Real vs Permuted Label SHAP Distribution')
ax.legend(fontsize=8)

# Panel 3: Prevalence vs Importance scatter
ax=axes[0,2]
sc=ax.scatter(all_prev,all_imp,c=all_imp,cmap='YlOrRd',alpha=0.5,s=30,edgecolors='none')
# Highlight novel candidates
novel_x=[g['prevalence'] for g in novel_candidates[:5]]
novel_y=[g['abs_importance'] for g in novel_candidates[:5]]
novel_names=[g['genus_name'] for g in novel_candidates[:5]]
ax.scatter(novel_x,novel_y,c='red',marker='*',s=200,edgecolors='black',linewidths=0.5)
for i in range(len(novel_x)):
    ax.annotate(novel_names[i],(novel_x[i],novel_y[i]),fontsize=7,xytext=(5,5),textcoords="offset points")
ax.set_xlabel('Prevalence'); ax.set_ylabel('|SHAP Importance|')
ax.set_title(f'Prevalence vs Importance (r={np.corrcoef(all_prev,all_imp)[0,1]:.3f})')
plt.colorbar(sc,ax=ax,shrink=0.7,label='|SHAP|')

# Panel 4: Cluster comparison
ax=axes[1,0]
metrics=['Richness','Dist2Healthy','Entropy']
c0_vals=[np.mean(c0_richness),np.mean(c0_dists),np.mean(c0_entropy)]
c1_vals=[np.mean(c1_richness),np.mean(c1_dists),np.mean(c1_entropy)]
x=np.arange(len(metrics)); w=0.35
ax.bar(x-w/2,c0_vals,w,label=f'Cluster 0 (n={len(c0_richness)})',color='#4CAF50',edgecolor='none')
ax.bar(x+w/2,c1_vals,w,label=f'Cluster 1 (n={len(c1_richness)})',color='#F44336',edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title('IBD Cluster Comparison'); ax.legend(fontsize=8)
ax.grid(True,alpha=0.3,axis='y')

# Panel 5: Abundance rank histogram for top SHAP genera
ax=axes[1,1]
# Split genera into abundance bins
low_rank=[g['mean_abundance_rank'] for g in abundance_data if g['abs_importance']>np.percentile(all_imp,75)]
high_rank=[g['mean_abundance_rank'] for g in abundance_data if g['abs_importance']<=np.percentile(all_imp,25)]
ax.hist(low_rank,bins=20,alpha=0.6,label=f'High |SHAP| (P75+)',color='#F44336')
ax.hist(high_rank,bins=20,alpha=0.6,label=f'Low |SHAP| (P25-)',color='#4CAF50')
ax.set_xlabel('Mean Abundance Rank (0=most abundant)'); ax.set_ylabel('Frequency')
ax.set_title('Abundance Rank by SHAP Importance')
ax.legend(fontsize=8)

# Panel 6: Summary
ax=axes[1,2]
ax.axis('off')
msg=f"""Phase 3.6: Biomarker Robustness

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CV SHAP Stability:
  Top-10 Jaccard: {jaccard_results[10]['mean']:.3f} ± {jaccard_results[10]['std']:.3f}
  Top-20 Jaccard: {jaccard_results[20]['mean']:.3f} ± {jaccard_results[20]['std']:.3f}
  Top-50 Jaccard: {jaccard_results[50]['mean']:.3f} ± {jaccard_results[50]['std']:.3f}

Permutation Control:
  Real |SHAP|: {np.mean(real_abs_imps):.5f} (max={np.max(real_abs_imps):.4f})
  Perm |SHAP|: {np.mean(perm_abs_imps):.5f} (max={np.max(perm_abs_imps):.4f})
  Ratio: {np.mean(real_abs_imps)/np.mean(perm_abs_imps):.1f}x

Abundance-Rarity:
  Corr(prev,|SHAP|): {np.corrcoef(all_prev,all_imp)[0,1]:.3f}
  Top-20 SHAP mean prev: {np.mean(top20_prev):.3f}
  Novel candidates (low prev, high SHAP): {len(novel_candidates)}

Cluster Validation:
  Cluster 1 (6%): distinct from C0 in:
  - Richness: {np.mean(c1_richness):.0f} vs {np.mean(c0_richness):.0f}
  - Distance: {np.mean(c1_dists):.3f} vs {np.mean(c0_dists):.3f}
  - Entropy:  {np.mean(c1_entropy):.3f} vs {np.mean(c0_entropy):.3f}
{f'  Cluster 1 is a DISTINCT IBD subtype' if np.mean(c1_dists)>np.mean(c0_dists)*2 else f'  Cluster 1: {np.mean(c1_dists)/np.mean(c0_dists):.1f}x farther from Healthy'}
"""
ax.text(0.05,0.95,msg,transform=ax.transAxes,fontsize=9.5,verticalalignment='top',fontfamily='monospace',
    bbox=dict(boxstyle='round',facecolor='#F5F5F5',alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/phase36_robustness.png',dpi=150,bbox_inches='tight')
print(f"Saved: {OUT_DIR}/phase36_robustness.png")

# Save report
robustness_report={
    'cv_shap_jaccard':jaccard_results,
    'consistent_genera_across_folds':{
        'top20':list(fold_topk[20][0].intersection(*fold_topk[20][1:])),
        'top50':list(fold_topk[50][0].intersection(*fold_topk[50][1:]))
    },
    'permutation_control':{
        'real_mean_abs_shap':float(np.mean(real_abs_imps)),
        'perm_mean_abs_shap':float(np.mean(perm_abs_imps)),
        'ratio':float(np.mean(real_abs_imps)/np.mean(perm_abs_imps)),
        'real_top20':list(real_top20),
        'perm_top20':list(perm_top20)
    },
    'abundance_importance_correlation':float(np.corrcoef(all_prev,all_imp)[0,1]),
    'novel_candidates':[{'genus_name':g['genus_name'],'abs_importance':g['abs_importance'],
        'prevalence':g['prevalence'],'mean_abundance_rank':g['mean_abundance_rank']}
        for g in novel_candidates[:20]],
    'cluster_validation':{
        'c0':{'richness':float(np.mean(c0_richness)),'dist_to_healthy':float(np.mean(c0_dists)),'entropy':float(np.mean(c0_entropy))},
        'c1':{'richness':float(np.mean(c1_richness)),'dist_to_healthy':float(np.mean(c1_dists)),'entropy':float(np.mean(c1_entropy))}
    }
}

with open(f'{OUT_DIR}/robustness_report.json','w') as f:
    json.dump(robustness_report,f,indent=2,default=str)
print(f"Saved: {OUT_DIR}/robustness_report.json")
print(f"\n{'='*60}")
print("PHASE 3.6 COMPLETE")
print(f"{'='*60}")
