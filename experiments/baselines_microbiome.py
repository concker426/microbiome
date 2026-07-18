#!/usr/bin/env python3
"""Microbiome-specific baselines for IBD classification on clean_2538.

Baselines:
  A: Raw abundance vector + XGBoost               (from H1.1)
  B: Raw abundance vector + MLP                    (from H1.1)
  C: Simple genus embedding + mean pool + MLP      (NEW - tests embedding value)
  D: MGM pretrained 768-dim + MLP                  (from H1.2b)

Baseline C uses the SAME embedding dimension (768) as MGM but with simple
mean pooling over genus embeddings (no Transformer, no pretraining).
This isolates whether the MGM Transformer architecture + pretraining
matters beyond simple embedding.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
from run_v6_merged import MGMEnc

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
ENCODER_PATH = '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)

def load_data():
    data = []
    for split in ['train_nl.jsonl', 'test_nl.jsonl']:
        with open(os.path.join(DATA_DIR, split)) as f:
            for line in f: data.append(json.loads(line))
    train_seqs = np.load(os.path.join(DATA_DIR, 'train_genus_sequences.npy'))
    test_seqs = np.load(os.path.join(DATA_DIR, 'test_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA_DIR, 'train_genus_masks.npy'))
    test_masks = np.load(os.path.join(DATA_DIR, 'test_genus_masks.npy'))
    all_seqs = np.concatenate([train_seqs, test_seqs], axis=0)
    all_masks = np.concatenate([train_masks, test_masks], axis=0)
    all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in data])
    return all_seqs, all_masks, all_labels

def extract_simple_embedding(seqs, masks, embed_dim=768):
    """Simple genus embedding + masked mean pooling (no Transformer, no pretraining).
    Same vocab size and embed dim as MGM for fair comparison.
    """
    vocab_size = 1226
    # Random init embedding (same as MGM would have without pretraining)
    emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    emb.to(device)

    feats = []
    for i in range(0, len(seqs), 128):
        gi = torch.from_numpy(seqs[i:i+128].astype(np.int64)).long().to(device)
        gm = torch.from_numpy(masks[i:i+128]).bool().to(device)
        with torch.no_grad():
            e = emb(gi)  # [B, SL, 768]
            # Masked mean pooling
            gm_float = gm.float().unsqueeze(-1)  # [B, SL, 1]
            pooled = (e * gm_float).sum(dim=1) / gm_float.sum(dim=1).clamp(min=1)  # [B, 768]
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)

def extract_mgm_features(seqs, masks):
    enc = MGMEnc()
    ck = torch.load(ENCODER_PATH, map_location='cpu')
    st = ck.get('model_state_dict', ck)
    enc.load_state_dict(st, strict=False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    enc.to(device).eval()
    feats = []
    for i in range(0, len(seqs), 64):
        gi = torch.from_numpy(seqs[i:i+64].astype(np.int64)).long().to(device)
        gm = torch.from_numpy(masks[i:i+64]).bool().to(device)
        with torch.no_grad():
            feats.append(enc(gi, gm).cpu().numpy())
    return np.concatenate(feats, axis=0)

def evaluate_all(X, y, name):
    classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    }
    results = {}
    for clf_name, clf in classifiers.items():
        all_scores = []
        for seed in range(3):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+seed)
            scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
            all_scores.extend(scores)
        results[clf_name] = {'mean': float(np.mean(all_scores)), 'std': float(np.std(all_scores))}
        best_clf = clf_name
    best = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"  {name}: best={best[0]} ACC={best[1]['mean']:.4f}±{best[1]['std']:.4f}")
    return results, best

def main():
    print("=" * 60)
    print("Microbiome Baselines: IBD Classification (clean_2538)")
    print("=" * 60)

    all_seqs, all_masks, all_labels = load_data()
    print(f"  {len(all_labels)} samples, Disease={all_labels.sum()}")

    all_bests = {}

    # A: Raw abundance vector (86-dim)
    print("\n--- A: Raw genus IDs (86-dim) ---")
    raw = all_seqs.astype(np.float32) / 1226.0
    ra, ba = evaluate_all(raw, all_labels, "Raw-86")
    all_bests['A_Raw_XGBoost'] = ba

    # B: Will be same classifiers on raw, but best MLP
    all_bests['B_Raw_MLP'] = ('MLP', ra['MLP']['mean'])

    # Baseline E: Simple embedding is a separate strong baseline
    # Will be filled in after C evaluates

    # C: Simple embedding + mean pool (768-dim, random init)
    print("\n--- C: Simple Embedding + Mean Pool (768-dim, random init) ---")
    t0 = time.time()
    simple = extract_simple_embedding(all_seqs, all_masks, 768)
    print(f"  Extraction: {time.time()-t0:.1f}s, shape={simple.shape}")
    rc, bc = evaluate_all(simple, all_labels, "SimpleEmb-768")
    all_bests['C_SimpleEmb_MLP'] = bc

    # D: MGM pretrained (768-dim)
    print("\n--- D: MGM Pretrained Transformer (768-dim) ---")
    t0 = time.time()
    mgm = extract_mgm_features(all_seqs, all_masks)
    print(f"  Extraction: {time.time()-t0:.1f}s, shape={mgm.shape}")
    rd, bd = evaluate_all(mgm, all_labels, "MGM-768")
    all_bests['D_MGM_MLP'] = bd

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<45} {'Best CLF':<20} {'ACC':>10}")
    print("-" * 77)
    for name, (clf, acc) in all_bests.items():
        print(f"{name:<45} {clf:<20} {acc:>10.4f}")

    output = {
        'experiment': 'microbiome_baselines',
        'baselines': {name: {'classifier': clf, 'accuracy': acc} for name, (clf, acc) in all_bests.items()},
        'raw_results': {'A_raw': ra, 'C_simple_emb': rc, 'D_mgm': rd},
        'timestamp': str(__import__('datetime').datetime.now()),
    }
    # Key comparison
    if 'D_MGM_MLP' in all_bests and 'C_SimpleEmb_MLP' in all_bests:
        mgm_acc = all_bests['D_MGM_MLP'][1]
        simple_acc = all_bests['C_SimpleEmb_MLP'][1]
        output['metrics'] = {
            'mgm_vs_simple_embedding': mgm_acc - simple_acc,
            'mgm_vs_raw': mgm_acc - all_bests['A_Raw_XGBoost'][1],
        }
        print(f"\nMGM vs Simple Embedding: {mgm_acc-simple_acc:+.4f}")
        print(f"MGM vs Raw XGBoost: {mgm_acc-all_bests['A_Raw_XGBoost'][1]:+.4f}")

    with open(os.path.join(RESULT_DIR, 'baselines.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/baselines.json")

if __name__ == '__main__':
    main()
