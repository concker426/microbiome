#!/usr/bin/env python3
"""Phase 3.5: IBD Heterogeneity — embedding clustering + subtype SHAP patterns"""
import sys,os,csv,json,pickle
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, f_oneway

OUT_DIR='/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR,exist_ok=True)

# ── Load data ──
emb=np.load(f'{OUT_DIR}/../backbone/embeddings.npy')
labels=np.load(f'{OUT_DIR}/../backbone/labels.npy')
ids=np.load(f'{OUT_DIR}/../backbone/sample_ids.npy')

# Load predictions
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_rows=list(csv.DictReader(f))
# Load SHAP sample data
with open(f'{OUT_DIR}/shap_data_full.pkl','rb') as f:
    shap_data=pickle.load(f)

# Build sample_id → SHAP lookup
shap_by_id={}
for s in shap_data['all_samples']:
    shap_by_id[s['sample_id']]={'label':s['label'],'importance':s['importance']}

# Split train/test
train_ids_set=set(r['sample_id'] for r in pred_rows if r['split']=='train')
test_ids_set=set(r['sample_id'] for r in pred_rows if r['split']=='test')
train_mask=np.array([sid in train_ids_set for sid in ids])
test_mask=np.array([sid in test_ids_set for sid in ids])

emb_train=emb[train_mask]; labels_train=labels[train_mask]; ids_train=ids[train_mask]
emb_test=emb[test_mask]; labels_test=labels[test_mask]; ids_test=ids[test_mask]

# Disease samples in train
disease_train_mask=labels_train==1
emb_disease=emb_train[disease_train_mask]; ids_disease=ids_train[disease_train_mask]
print(f"IBD samples in train: {len(emb_disease)}")

# ── 1. Determine optimal K for IBD clustering ──
print(f"\n{'='*60}")
print("1. IBD Clustering — Optimal K")
print(f"{'='*60}")

K_range=range(2,min(8,len(emb_disease)-1))
sil_scores=[]
for k in K_range:
    km=KMeans(n_clusters=k,random_state=42,n_init=20)
    labels_k=km.fit_predict(emb_disease)
    if len(set(labels_k))>1:
        sil=silhouette_score(emb_disease,labels_k,metric='cosine',random_state=42)
        sil_scores.append(sil)
        print(f"  K={k}: Silhouette={sil:.4f} sizes={np.bincount(labels_k[labels_k>=0])}")
    else:
        sil_scores.append(-1)
        print(f"  K={k}: single cluster")

best_k=K_range[np.argmax(sil_scores)]
print(f"  Best K={best_k} (silhouette={max(sil_scores):.4f})")

# Fit with best K
km=KMeans(n_clusters=best_k,random_state=42,n_init=20)
cluster_labels=km.fit_predict(emb_disease)
cluster_sizes=np.bincount(cluster_labels[cluster_labels>=0])
print(f"  Cluster sizes: {cluster_sizes}")

# ── 2. Per-cluster SHAP analysis ──
print(f"\n{'='*60}")
print("2. Per-Cluster SHAP Profiles")
print(f"{'='*60}")

cluster_shap={c:{'genera':{},'sample_ids':[]} for c in range(best_k)}
for i,c in enumerate(cluster_labels):
    if c<0: continue
    sid=ids_disease[i]
    cluster_shap[c]['sample_ids'].append(sid)
    if sid in shap_by_id:
        for g in shap_by_id[sid]['importance']:
            gname=g['genus_name']
            if gname not in cluster_shap[c]['genera']:
                cluster_shap[c]['genera'][gname]=[]
            cluster_shap[c]['genera'][gname].append(g['importance'])

# Top genera per cluster
for c in range(best_k):
    genera=cluster_shap[c]['genera']
    if not genera: continue
    # Aggregate: mean importance across patients
    agg=[]
    for gname,imps in genera.items():
        agg.append((gname,np.mean(imps),len(imps)))
    agg.sort(key=lambda x:abs(x[1]),reverse=True)
    print(f"\n  Cluster {c} (n={cluster_sizes[c]}):")
    for gname,imp,n in agg[:10]:
        direction='↑IBD' if imp>0 else '↓IBD'
        print(f"    {gname:30s} imp={imp:+.4f} [{direction}] n={n}")

# ── 3. Cluster separation analysis ──
print(f"\n{'='*60}")
print("3. Cluster Separation")
print(f"{'='*60}")

centroids=km.cluster_centers_
# Pairwise cosine distances between centroids
for i in range(best_k):
    for j in range(i+1,best_k):
        ci=centroids[i]/np.linalg.norm(centroids[i])
        cj=centroids[j]/np.linalg.norm(centroids[j])
        cos_sim=ci@cj
        print(f"  Cluster {i} vs Cluster {j}: cosine similarity = {cos_sim:.4f} (distance = {1-cos_sim:.4f})")

# ── 4. Healthy reference comparison ──
print(f"\n{'='*60}")
print("4. Cluster vs Healthy Reference")
print(f"{'='*60}")

emb_healthy=emb_train[labels_train==0]
centroid_healthy=emb_healthy.mean(0)
centroid_healthy_n=centroid_healthy/np.linalg.norm(centroid_healthy)

for c in range(best_k):
    cc=centroids[c]/np.linalg.norm(centroids[c])
    cos_sim=cc@centroid_healthy_n
    print(f"  Cluster {c} (n={cluster_sizes[c]}): cosine similarity to Healthy = {cos_sim:.4f}")

# ── 5. Statistical test: are clusters significantly different? ──
print(f"\n{'='*60}")
print("5. Cluster Distinctiveness")
print(f"{'='*60}")

# For each patient group, compute distribution of distances to its own centroid vs others
# Use the SHAP top-5 genera per cluster for ANOVA
all_top_genera=set()
for c in range(best_k):
    genera=cluster_shap[c]['genera']
    agg=[(gname,np.mean(imps)) for gname,imps in genera.items()]
    agg.sort(key=lambda x:abs(x[1]),reverse=True)
    all_top_genera.update([g[0] for g in agg[:5]])

print(f"  Top genera across clusters: {all_top_genera}")

# Per-sample: which SHAP features differ most between clusters?
# For each pair of clusters, do Mann-Whitney on top-genera importance distributions
for ci in range(best_k):
    for cj in range(ci+1,best_k):
        diffs=[]
        for gn in list(all_top_genera)[:10]:
            imps_i=[g['importance'] for sid in cluster_shap[ci]['sample_ids']
                    if sid in shap_by_id
                    for g in shap_by_id[sid]['importance']
                    if g['genus_name']==gn]
            imps_j=[g['importance'] for sid in cluster_shap[cj]['sample_ids']
                    if sid in shap_by_id
                    for g in shap_by_id[sid]['importance']
                    if g['genus_name']==gn]
            if len(imps_i)>=3 and len(imps_j)>=3:
                try:
                    stat,p=mannwhitneyu(imps_i,imps_j,alternative='two-sided')
                    if p<0.05:
                        diffs.append((gn,p,abs(np.mean(imps_i)-np.mean(imps_j))))
                except: pass
        diffs.sort(key=lambda x:x[2],reverse=True)
        if diffs:
            print(f"  Cluster {ci} vs {cj} — significantly different genera:")
            for gn,p,d in diffs[:5]:
                print(f"    {gn:30s} p={p:.4f} diff={d:.4f}")

# ── 6. Check: do clusters correspond to clinical metadata? ──
# (no clinical metadata available for clean_2538)
# Instead: check if clusters differ in prediction confidence

print(f"\n{'='*60}")
print("6. Cluster vs Prediction Confidence")
print(f"{'='*60}")

for c in range(best_k):
    probs=[]
    for sid in cluster_shap[c]['sample_ids']:
        for r in pred_rows:
            if r['sample_id']==sid:
                probs.append(float(r['prob_disease']))
                break
    if probs:
        print(f"  Cluster {c} (n={len(probs)}): mean prob_disease = {np.mean(probs):.4f} ± {np.std(probs):.4f}")

# ── Visualization ──
# Get UMAP coords for train disease
umap_coords=np.load(f'{OUT_DIR}/umap_coords.npy')
umap_labels_all=np.load(f'{OUT_DIR}/umap_labels.npy')
# Since UMAP was on all data, need to align with train set
all_ids_umap=np.load(f'{OUT_DIR}/../backbone/sample_ids.npy')

fig,axes=plt.subplots(2,2,figsize=(16,14))

# Panel 1: UMAP with IBD clusters
ax=axes[0,0]
# Map cluster labels to full UMAP
disease_indices=np.where((labels_train==1))[0]
cluster_map_full={ids_disease[i]:cluster_labels[i] for i in range(len(ids_disease))}

# Plot healthy first
h_mask=(labels==0)
ax.scatter(umap_coords[h_mask,0],umap_coords[h_mask,1],c='#BDBDBD',alpha=0.3,s=15,label='Healthy',edgecolors='none')

# Plot each cluster
cluster_colors=plt.cm.Set2(np.linspace(0,1,best_k))
for c in range(best_k):
    cluster_samples=set(cluster_shap[c]['sample_ids'])
    mask=np.array([sid in cluster_samples for sid in ids])
    ax.scatter(umap_coords[mask,0],umap_coords[mask,1],
        c=[cluster_colors[c]],alpha=0.7,s=25,edgecolors='none',
        label=f'IBD Cluster {c} (n={cluster_sizes[c]})')
ax.set_title(f'IBD Heterogeneity: {best_k} Clusters (cosine UMAP)'); ax.legend(fontsize=7)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

# Panel 2: Per-cluster top SHAP genera
ax=axes[1,0]
bar_data=[]; bar_colors=[]; bar_labels=[]
for c in range(best_k):
    genera=cluster_shap[c]['genera']
    agg=[(gname,np.mean(imps)) for gname,imps in genera.items()]
    agg.sort(key=lambda x:abs(x[1]),reverse=True)
    # Take top 3 per cluster
    for gname,imp in agg[:3]:
        bar_data.append(imp)
        bar_colors.append(cluster_colors[c])
        bar_labels.append(f'C{c}:{gname}')

y=range(len(bar_data))
ax.barh(y,bar_data,color=bar_colors,edgecolor='none')
ax.set_yticks(y); ax.set_yticklabels(bar_labels,fontsize=7)
ax.set_xlabel('Mean SHAP Importance'); ax.set_title('Top SHAP Genera by Cluster')
ax.axvline(x=0,color='black',linewidth=0.5); ax.invert_yaxis()

# Panel 3: Cluster sizes + silhouette
ax=axes[0,1]
ax.bar(range(best_k),cluster_sizes,color=cluster_colors,edgecolor='none')
for i,v in enumerate(cluster_sizes):
    ax.text(i,v+1,str(v),ha='center',fontsize=10)
ax.set_xlabel('Cluster'); ax.set_ylabel('N samples')
ax.set_title(f'IBD Cluster Sizes (K={best_k}, Sil={max(sil_scores):.3f})')
ax.set_xticks(range(best_k))

# Panel 4: Cluster similarity matrix
ax=axes[1,1]
sim_matrix=np.zeros((best_k+1,best_k+1))  # +1 for healthy centroid
# IBD clusters
for i in range(best_k):
    for j in range(best_k):
        ci=centroids[i]/np.linalg.norm(centroids[i])
        cj=centroids[j]/np.linalg.norm(centroids[j])
        sim_matrix[i,j]=ci@cj
# Healthy
for i in range(best_k):
    ci=centroids[i]/np.linalg.norm(centroids[i])
    sim_matrix[i,best_k]=ci@centroid_healthy_n
    sim_matrix[best_k,i]=ci@centroid_healthy_n
sim_matrix[best_k,best_k]=1.0

labels_mat=[f'C{i}' for i in range(best_k)]+['Healthy']
im=ax.imshow(sim_matrix,cmap='RdYlBu_r',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(best_k+1)); ax.set_xticklabels(labels_mat,fontsize=9)
ax.set_yticks(range(best_k+1)); ax.set_yticklabels(labels_mat,fontsize=9)
ax.set_title('Cluster Cosine Similarity Matrix')
for i in range(best_k+1):
    for j in range(best_k+1):
        ax.text(j,i,f'{sim_matrix[i,j]:.3f}',ha='center',va='center',fontsize=8)
plt.colorbar(im,ax=ax,shrink=0.8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/phase35_heterogeneity.png',dpi=150,bbox_inches='tight')
print(f"\nSaved: {OUT_DIR}/phase35_heterogeneity.png")

# ── Save cluster assignments ──
cluster_results={
    'best_k':int(best_k),
    'silhouette_scores':{f'K{k}':float(s) for k,s in zip(K_range,sil_scores)},
    'cluster_sizes':cluster_sizes.tolist(),
    'cluster_assignments':{},
    'cluster_top_genera':{}
}

for c in range(best_k):
    genera=cluster_shap[c]['genera']
    agg=[(gname,float(np.mean(imps)),len(imps)) for gname,imps in genera.items()]
    agg.sort(key=lambda x:abs(x[1]),reverse=True)
    cluster_results['cluster_top_genera'][f'cluster_{c}']=agg[:20]
    for sid in cluster_shap[c]['sample_ids']:
        cluster_results['cluster_assignments'][sid]=c

with open(f'{OUT_DIR}/heterogeneity_results.json','w') as f:
    json.dump(cluster_results,f,indent=2,default=str)
print(f"Saved: {OUT_DIR}/heterogeneity_results.json")

print(f"\n{'='*60}")
print("PHASE 3.5 COMPLETE")
print(f"{'='*60}")
print(f"  IBD clusters: {best_k}")
print(f"  Silhouette: {max(sil_scores):.4f}")
print(f"  Sizes: {cluster_sizes}")
