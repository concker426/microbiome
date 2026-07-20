#!/usr/bin/env python3
"""Priority 5: UMAP visualization of embeddings"""
import json,sys,os
sys.path.insert(0,'/hd/liujx/microbiome_llm_project')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

OUT_DIR='/hd/liujx/microbiome_llm_project/experiments/results/final_backbone/embeddings'

# Load
emb_train=np.load(f'{OUT_DIR}/train_embeddings.npy')
emb_test=np.load(f'{OUT_DIR}/test_embeddings.npy')
labels_train=np.load(f'{OUT_DIR}/train_labels.npy')
labels_test=np.load(f'{OUT_DIR}/test_labels.npy')

print(f"Train: {emb_train.shape} Test: {emb_test.shape}")

# Combine
all_emb=np.concatenate([emb_train,emb_test],axis=0)
all_labels=np.concatenate([labels_train,labels_test],axis=0)
all_split=np.array(['Train']*len(emb_train)+['Test']*len(emb_test))

# UMAP
reducer=umap.UMAP(n_components=2,random_state=42,n_neighbors=15,min_dist=0.1,metric='cosine')
embedding_2d=reducer.fit_transform(all_emb)
print(f"UMAP done: {embedding_2d.shape}")

# Plot by label
fig,axes=plt.subplots(1,2,figsize=(14,6))

ax=axes[0]
for lbl,color,name in [(0,'#2196F3','Healthy'),(1,'#F44336','Disease')]:
    mask=all_labels==lbl
    ax.scatter(embedding_2d[mask,0],embedding_2d[mask,1],c=color,label=name,alpha=0.6,s=20,edgecolors='none')
ax.set_title('UMAP by Disease Status'); ax.legend(); ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

ax=axes[1]
for lbl,color,name in [(0,'#4CAF50','Train'),(1,'#FF9800','Test')]:
    mask=np.array(all_split)==name
    ax.scatter(embedding_2d[mask,0],embedding_2d[mask,1],c=color,label=name,alpha=0.6,s=20,edgecolors='none')
ax.set_title('UMAP by Split'); ax.legend(); ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/umap_visualization.png',dpi=150,bbox_inches='tight')
print(f"Saved to {OUT_DIR}/umap_visualization.png")

# Save UMAP coordinates
np.save(f'{OUT_DIR}/umap_coords.npy',embedding_2d)
np.save(f'{OUT_DIR}/umap_labels.npy',all_labels)
np.save(f'{OUT_DIR}/umap_splits.npy',np.array(all_split))
print("UMAP coordinates saved.")
