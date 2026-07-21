#!/usr/bin/env python3
"""Phase 1: Representation Analysis — why SimpleEmb works"""
import sys,os,csv,json
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, pairwise_distances,
    accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import umap

OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR,exist_ok=True)

# ── Load data ──
emb=np.load(f'{OUT_DIR}/../backbone/embeddings.npy')
labels=np.load(f'{OUT_DIR}/../backbone/labels.npy')
ids=np.load(f'{OUT_DIR}/../backbone/sample_ids.npy')

# Load predictions for error analysis
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    reader=csv.DictReader(f)
    pred_rows=list(reader)
test_rows=[r for r in pred_rows if r['split']=='test']
train_rows=[r for r in pred_rows if r['split']=='train']

# Build split masks
train_ids_set=set(r['sample_id'] for r in train_rows)
test_ids_set=set(r['sample_id'] for r in test_rows)
train_mask=np.array([sid in train_ids_set for sid in ids])
test_mask=np.array([sid in test_ids_set for sid in ids])
# Build correct/error mask for test
test_correct=np.zeros(len(ids),dtype=bool)
for r in test_rows:
    idx=list(ids).index(r['sample_id'])
    test_correct[idx]=(r['correct']=='True')

N=len(emb); n_train=train_mask.sum(); n_test=test_mask.sum()
print(f"Loaded: {N} samples ({n_train} train, {n_test} test)")
print(f"  Healthy: {(labels==0).sum()}, Disease: {(labels==1).sum()}")

# ── Output collector ──
report={}

# ============================================================
# Experiment 1: PCA
# ============================================================
print("\n"+"="*60)
print("Experiment 1: PCA")
print("="*60)

pca=PCA(n_components=2,random_state=42)
emb_pca=pca.fit_transform(emb)
var_explained=pca.explained_variance_ratio_
print(f"PCA explained variance: {var_explained[0]:.4f}, {var_explained[1]:.4f} (sum={var_explained.sum():.4f})")
report['pca']={'explained_variance_ratio':var_explained.tolist(),'total':float(var_explained.sum())}

# For higher-dim PCA
pca_full=PCA(n_components=min(200,N),random_state=42)
pca_full.fit(emb)
cumsum=np.cumsum(pca_full.explained_variance_ratio_)
dims_90=np.where(cumsum>=0.90)[0]
dims_95=np.where(cumsum>=0.95)[0]
d90=int(dims_90[0]+1) if len(dims_90)>0 else len(cumsum)
d95=int(dims_95[0]+1) if len(dims_95)>0 else len(cumsum)
print(f"Dim for 90% var: {d90}, 95% var: {d95}")
report['pca']['dims_90pct']=d90; report['pca']['dims_95pct']=d95
report['pca']['total_var_50pc']=float(cumsum[min(49,len(cumsum)-1)])

# ============================================================
# Experiment 2: UMAP
# ============================================================
print("\n"+"="*60)
print("Experiment 2: UMAP")
print("="*60)

reducer=umap.UMAP(n_components=2,random_state=42,n_neighbors=15,min_dist=0.1,metric='cosine')
emb_umap=reducer.fit_transform(emb)

# ============================================================
# Experiment 3: t-SNE
# ============================================================
print("\n"+"="*60)
print("Experiment 3: t-SNE")
print("="*60)

tsne=TSNE(n_components=2,random_state=42,perplexity=30,metric='cosine',n_jobs=1)
emb_tsne=tsne.fit_transform(emb)
print("t-SNE done")

# ============================================================
# Experiment 4: Silhouette Score
# ============================================================
print("\n"+"="*60)
print("Experiment 4: Silhouette Score + Clustering")
print("="*60)

# Silhouette on raw embeddings (cosine distance)
# Subsample for speed if needed
n_subsample=min(800,N)
idx_sub=np.random.RandomState(42).choice(N,size=n_subsample,replace=False)
emb_sub=emb[idx_sub]; labels_sub=labels[idx_sub]

# Euclidean silhouette
sil_euc=silhouette_score(emb_sub,labels_sub,metric='euclidean',random_state=42)
# Cosine: compute pairwise manually
cos_dist=pairwise_distances(emb_sub,metric='cosine')
sil_cos=silhouette_score(cos_dist,labels_sub,metric='precomputed')
print(f"Silhouette (euclidean): {sil_euc:.4f}")
print(f"Silhouette (cosine):    {sil_cos:.4f}")
report['silhouette']={'euclidean':float(sil_euc),'cosine':float(sil_cos),'n_subsample':n_subsample}

# Also on UMAP and PCA spaces
sil_umap=silhouette_score(emb_umap[idx_sub],labels_sub,metric='euclidean',random_state=42)
sil_pca=silhouette_score(emb_pca[idx_sub],labels_sub,metric='euclidean',random_state=42)
sil_tsne=silhouette_score(emb_tsne[idx_sub],labels_sub,metric='euclidean',random_state=42)
print(f"Silhouette (UMAP space):  {sil_umap:.4f}")
print(f"Silhouette (PCA space):   {sil_pca:.4f}")
print(f"Silhouette (t-SNE space): {sil_tsne:.4f}")
report['silhouette_2d']={'umap':float(sil_umap),'pca':float(sil_pca),'tsne':float(sil_tsne)}

# ============================================================
# Experiment 5: Cosine Similarity Analysis
# ============================================================
print("\n"+"="*60)
print("Experiment 5: Cosine Similarity")
print("="*60)

# Normalize for cosine
emb_norm=emb/np.linalg.norm(emb,axis=1,keepdims=True).clip(min=1e-8)

h_mask=labels==0; d_mask=labels==1
emb_h=emb_norm[h_mask]; emb_d=emb_norm[d_mask]

# Intra-class cosine similarity
cos_hh=(emb_h @ emb_h.T).mean()
cos_dd=(emb_d @ emb_d.T).mean()
# Inter-class
cos_hd=(emb_h @ emb_d.T).mean()

print(f"Cosine similarity (Healthy-Healthy):  {cos_hh:.4f}")
print(f"Cosine similarity (Disease-Disease):  {cos_dd:.4f}")
print(f"Cosine similarity (Healthy-Disease):  {cos_hd:.4f}")
print(f"Separation gap: {(cos_hh+cos_dd)/2 - cos_hd:.4f}")
report['cosine']={
    'intra_healthy':float(cos_hh),'intra_disease':float(cos_dd),
    'inter_class':float(cos_hd),
    'separation_gap':float((cos_hh+cos_dd)/2-cos_hd)
}

# ============================================================
# Experiment 6: kNN Accuracy
# ============================================================
print("\n"+"="*60)
print("Experiment 6: kNN Accuracy")
print("="*60)

# kNN on test set only (train as reference)
emb_train=emb[train_mask]; labels_train=labels[train_mask]
emb_test=emb[test_mask]; labels_test=labels[test_mask]

knn_results={}
for k in [1,3,5,10,15,30]:
    knn=KNeighborsClassifier(n_neighbors=k,metric='cosine',weights='distance')
    knn.fit(emb_train,labels_train)
    pred=knn.predict(emb_test)
    acc=accuracy_score(labels_test,pred)
    prec,rec,f1,_=precision_recall_fscore_support(labels_test,pred,average='macro')
    cm=confusion_matrix(labels_test,pred)
    knn_results[f'k{k}']={'acc':float(acc),'macro_f1':float(f1),'cm':cm.tolist()}
    print(f"  k={k:2d}: ACC={acc:.4f}  F1={f1:.4f}  CM={cm.ravel()}")

# 5-fold CV on train for kNN
best_k=5
knn_cv_accs=[]
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for fold,(tr_idx,val_idx) in enumerate(skf.split(emb_train,labels_train)):
    knn=KNeighborsClassifier(n_neighbors=best_k,metric='cosine',weights='distance')
    knn.fit(emb_train[tr_idx],labels_train[tr_idx])
    pred=knn.predict(emb_train[val_idx])
    acc=accuracy_score(labels_train[val_idx],pred)
    knn_cv_accs.append(acc)
    print(f"  kNN(k={best_k}) Fold {fold}: ACC={acc:.4f}")
print(f"  kNN(k={best_k}) CV: {np.mean(knn_cv_accs):.4f} ± {np.std(knn_cv_accs):.4f}")
report['knn']={'results':knn_results,'cv_mean':float(np.mean(knn_cv_accs)),'cv_std':float(np.std(knn_cv_accs))}

# ============================================================
# Experiment 7: Error Analysis
# ============================================================
print("\n"+"="*60)
print("Experiment 7: Error Analysis")
print("="*60)

test_correct_mask=test_correct[test_mask]
test_labels_local=labels[test_mask]
test_emb_local=emb[test_mask]

fp_mask=(test_correct_mask==False)&(test_labels_local==0)  # Healthy predicted Disease
fn_mask=(test_correct_mask==False)&(test_labels_local==1)  # Disease predicted Healthy

n_fp=fp_mask.sum(); n_fn=fn_mask.sum()
print(f"False Positives (Healthy→Disease): {n_fp}")
print(f"False Negatives (Disease→Healthy): {n_fn}")

# Distance to class centroids for errors vs correct
centroid_h=emb_train[labels_train==0].mean(0)
centroid_d=emb_train[labels_train==1].mean(0)

def cos_dist(a,b):
    a_n=a/np.linalg.norm(a,axis=-1,keepdims=True).clip(min=1e-8)
    b_n=b/np.linalg.norm(b,keepdims=True).clip(min=1e-8)
    return 1-(a_n @ b_n)

# For correct Healthy predictions
correct_h=test_emb_local[(test_correct_mask==True)&(test_labels_local==0)]
# For FP (Healthy→Disease)
fp_emb=test_emb_local[fp_mask]
# For correct Disease predictions
correct_d=test_emb_local[(test_correct_mask==True)&(test_labels_local==1)]
# For FN (Disease→Healthy)
fn_emb=test_emb_local[fn_mask]

for name,embs in [('Correct_H',correct_h),('FP (H→D)',fp_emb),('Correct_D',correct_d),('FN (D→H)',fn_emb)]:
    if len(embs)>0:
        d_h=cos_dist(embs,centroid_h)
        d_d=cos_dist(embs,centroid_d)
        print(f"  {name:15s}: dist2Healthy={d_h.mean():.4f}, dist2Disease={d_d.mean():.4f}, diff={d_h.mean()-d_d.mean():.4f}")

report['error_analysis']={
    'n_fp':int(n_fp),'n_fn':int(n_fn),
    'test_total':int(n_test)
}

# ============================================================
# Experiment 8: Batch Effect (Train vs Test)
# ============================================================
print("\n"+"="*60)
print("Experiment 8: Batch Effect (Train vs Test)")
print("="*60)

# Check if train and test overlap in embedding space
train_h_mask=train_mask&(labels==0); test_h_mask=test_mask&(labels==0)
train_d_mask=train_mask&(labels==1); test_d_mask=test_mask&(labels==1)

for name,m1,m2 in [
    ('Train_H vs Test_H',train_h_mask,test_h_mask),
    ('Train_D vs Test_D',train_d_mask,test_d_mask),
    ('Train_H vs Train_D',train_h_mask,train_d_mask),
]:
    c1=emb_norm[m1].mean(0); c2=emb_norm[m2].mean(0)
    dist=1-(c1@c2)
    print(f"  {name}: cosine distance between centroids = {dist:.4f}")

# Overlap metric: for each test sample, check if nearest neighbor is from train
knn_batch=KNeighborsClassifier(n_neighbors=1,metric='cosine')
knn_batch.fit(emb_train,labels_train)
nearest_labels=knn_batch.predict(emb_test)
same_label=(nearest_labels==labels_test).mean()
print(f"  Test samples with same-label nearest train neighbor: {same_label:.4f}")

report['batch_effect']={
    'same_label_nn':float(same_label)
}

# ============================================================
# Visualization
# ============================================================
print("\n"+"="*60)
print("Generating figures...")
print("="*60)

fig,axes=plt.subplots(2,3,figsize=(18,12))
colors={0:'#2196F3',1:'#F44336'}
names={0:'Healthy',1:'Disease'}

# PCA
ax=axes[0,0]
for lbl in [0,1]:
    m=labels==lbl
    ax.scatter(emb_pca[m,0],emb_pca[m,1],c=colors[lbl],label=names[lbl],alpha=0.5,s=15,edgecolors='none')
ax.set_title(f'PCA (var={var_explained.sum():.3f})'); ax.legend(fontsize=8); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

# UMAP
ax=axes[0,1]
for lbl in [0,1]:
    m=labels==lbl
    ax.scatter(emb_umap[m,0],emb_umap[m,1],c=colors[lbl],label=names[lbl],alpha=0.5,s=15,edgecolors='none')
ax.set_title(f'UMAP (sil={sil_umap:.3f})'); ax.legend(fontsize=8); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

# t-SNE
ax=axes[0,2]
for lbl in [0,1]:
    m=labels==lbl
    ax.scatter(emb_tsne[m,0],emb_tsne[m,1],c=colors[lbl],label=names[lbl],alpha=0.5,s=15,edgecolors='none')
ax.set_title(f't-SNE (sil={sil_tsne:.3f})'); ax.legend(fontsize=8); ax.set_xlabel('tSNE1'); ax.set_ylabel('tSNE2')

# Error analysis on UMAP (test set only)
ax=axes[1,0]
for lbl in [0,1]:
    m=(labels==lbl)&test_mask&test_correct
    ax.scatter(emb_umap[m,0],emb_umap[m,1],c=colors[lbl],label=f'{names[lbl]} (correct)',alpha=0.5,s=20,edgecolors='none',marker='o')
# FP and FN
m_fp=test_mask&(test_correct==False)&(labels==0)
m_fn=test_mask&(test_correct==False)&(labels==1)
if m_fp.sum()>0:
    ax.scatter(emb_umap[m_fp,0],emb_umap[m_fp,1],c='red',marker='x',s=80,linewidths=2,label=f'FP (H→D, n={m_fp.sum()})')
if m_fn.sum()>0:
    ax.scatter(emb_umap[m_fn,0],emb_umap[m_fn,1],c='blue',marker='+',s=80,linewidths=2,label=f'FN (D→H, n={m_fn.sum()})')
ax.set_title('Error Analysis (UMAP)'); ax.legend(fontsize=7); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

# Cosine similarity distribution
ax=axes[1,1]
# Sample pairs for histogram
np.random.seed(42)
rng=np.random.RandomState(42)
h_idx=np.where(h_mask)[0]; d_idx=np.where(d_mask)[0]
n_pairs=2000
# Intra healthy
pairs_hh=emb_norm[h_idx[rng.choice(len(h_idx),n_pairs)]] @ emb_norm[h_idx[rng.choice(len(h_idx),n_pairs)]].T
hh_diag=np.diag(pairs_hh)
# Intra disease
pairs_dd=emb_norm[d_idx[rng.choice(len(d_idx),n_pairs)]] @ emb_norm[d_idx[rng.choice(len(d_idx),n_pairs)]].T
dd_diag=np.diag(pairs_dd)
# Inter
pairs_hd=emb_norm[h_idx[rng.choice(len(h_idx),n_pairs)]] @ emb_norm[d_idx[rng.choice(len(d_idx),n_pairs)]].T
hd_diag=np.diag(pairs_hd)

ax.hist(hh_diag,bins=50,alpha=0.5,label=f'Healthy-Healthy (mean={hh_diag.mean():.3f})',color='blue')
ax.hist(dd_diag,bins=50,alpha=0.5,label=f'Disease-Disease (mean={dd_diag.mean():.3f})',color='red')
ax.hist(hd_diag,bins=50,alpha=0.5,label=f'Healthy-Disease (mean={hd_diag.mean():.3f})',color='gray')
ax.set_title('Cosine Similarity Distribution'); ax.legend(fontsize=7); ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Frequency')

# kNN accuracy vs k
ax=axes[1,2]
ks=[1,3,5,10,15,30]
accs=[knn_results[f'k{k}']['acc'] for k in ks]
ax.plot(ks,accs,'o-',color='#4CAF50',linewidth=2,markersize=8)
ax.axhline(y=0.9281,color='gray',linestyle='--',label=f'MLP ensemble (0.928)')
ax.set_xlabel('k'); ax.set_ylabel('ACC'); ax.set_title('kNN Accuracy vs k'); ax.legend(fontsize=8)
ax.set_xlim(0,max(ks)+1)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/phase1_representation_analysis.png',dpi=150,bbox_inches='tight')
print(f"Saved: {OUT_DIR}/phase1_representation_analysis.png")

# ── Save report ──
report['dataset']={'n_samples':N,'n_train':int(n_train),'n_test':int(n_test),
    'n_healthy':int((labels==0).sum()),'n_disease':int((labels==1).sum()),
    'emb_dim':emb.shape[1]}

with open(f'{OUT_DIR}/phase1_report.json','w') as f:
    json.dump(report,f,indent=2,default=str)
print(f"Saved: {OUT_DIR}/phase1_report.json")

# ── Save coordinates ──
np.save(f'{OUT_DIR}/pca_coords.npy',emb_pca)
np.save(f'{OUT_DIR}/umap_coords.npy',emb_umap)
np.save(f'{OUT_DIR}/tsne_coords.npy',emb_tsne)

print("\n"+"="*60)
print("PHASE 1 DONE")
print("="*60)
