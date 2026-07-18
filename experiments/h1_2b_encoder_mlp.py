#!/usr/bin/env python3
"""H1.2b: Pretrained MGM vs Random MGM — pure representation test (NO LLM, NO dropout).

Extracts 768-dim features from MGM encoder (pretrained vs random init),
trains MLP classifier on clean_2538. 5-fold CV with 5 repeats.
Answers: does MGM PRETRAINING produce better representations than random init?
"""
import json, os, sys, time
import numpy as np
import torch
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

def extract_features(seqs, masks, encoder_state='pretrained'):
    """Extract 768-dim features. encoder_state: 'pretrained' or 'random'."""
    enc = MGMEnc()
    if encoder_state == 'pretrained':
        ck = torch.load(ENCODER_PATH, map_location='cpu')
        st = ck.get('model_state_dict', ck)
        enc.load_state_dict(st, strict=False)
    # else: random init (default)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    enc.to(device).eval()
    feats = []
    for i in range(0, len(seqs), 64):
        gi = torch.from_numpy(seqs[i:i+64].astype(np.int64)).long().to(device)
        gm = torch.from_numpy(masks[i:i+64]).bool().to(device)
        with torch.no_grad():
            feats.append(enc(gi, gm).cpu().numpy())
    return np.concatenate(feats, axis=0)

def evaluate(X, y, name, n_folds=5, n_repeats=3):
    """Stratified K-fold CV with multiple repeats."""
    classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    }
    results = {}
    for clf_name, clf in classifiers.items():
        all_scores = []
        for seed in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42+seed)
            scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
            all_scores.extend(scores)
        results[clf_name] = {
            'accuracy_mean': float(np.mean(all_scores)),
            'accuracy_std': float(np.std(all_scores)),
            'n_splits': len(all_scores),
        }
    return results

def main():
    print("=" * 60)
    print("H1.2b: Pretrained vs Random MGM (MLP only, no LLM, no dropout)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    all_seqs, all_masks, all_labels = load_data()
    n_samples = len(all_labels)
    print(f"  {n_samples} samples, Disease={all_labels.sum()} ({all_labels.sum()/n_samples*100:.1f}%)")

    print("\n[2/4] Extracting PRETRAINED MGM features...")
    t0 = time.time()
    pretrained_feats = extract_features(all_seqs, all_masks, 'pretrained')
    print(f"  {pretrained_feats.shape} ({time.time()-t0:.1f}s)")

    print("\n[3/4] Extracting RANDOM MGM features...")
    t0 = time.time()
    random_feats = extract_features(all_seqs, all_masks, 'random')
    print(f"  {random_feats.shape} ({time.time()-t0:.1f}s)")

    print("\n[4/4] Evaluating both...")
    print("\n--- Pretrained MGM ---")
    pretrained_results = evaluate(pretrained_feats, all_labels, "Pretrained")
    print("\n--- Random MGM ---")
    random_results = evaluate(random_feats, all_labels, "Random")

    # Summary
    print("\n" + "=" * 70)
    print("H1.2b RESULTS: Pretrained vs Random MGM Representations")
    print("=" * 70)
    print(f"{'Classifier':<20} {'Pretrained':>20} {'Random':>20} {'Δ':>10}")
    print("-" * 72)
    for clf_name in pretrained_results:
        p = pretrained_results[clf_name]['accuracy_mean']
        r = random_results[clf_name]['accuracy_mean']
        delta = p - r
        sign = '+' if delta > 0 else ''
        print(f"{clf_name:<20} {p:>19.4f} ±{pretrained_results[clf_name]['accuracy_std']:.4f}  "
              f"{r:>19.4f} ±{random_results[clf_name]['accuracy_std']:.4f}  {sign}{delta:>9.4f}")

    best_p = max(pretrained_results.items(), key=lambda x: x[1]['accuracy_mean'])
    best_r = max(random_results.items(), key=lambda x: x[1]['accuracy_mean'])
    gain = best_p[1]['accuracy_mean'] - best_r[1]['accuracy_mean']

    output = {
        'experiment': 'H1.2b',
        'hypothesis': 'MGM pretraining produces better representations than random init',
        'pretrained': pretrained_results,
        'random': random_results,
        'best_pretrained': {'classifier': best_p[0], 'accuracy': best_p[1]['accuracy_mean']},
        'best_random': {'classifier': best_r[0], 'accuracy': best_r[1]['accuracy_mean']},
        'pretraining_gain': gain,
        'timestamp': str(__import__('datetime').datetime.now()),
        'metrics': {'pretraining_gain': gain, 'pretrained_best_acc': best_p[1]['accuracy_mean'],
                     'random_best_acc': best_r[1]['accuracy_mean']}
    }
    with open(os.path.join(RESULT_DIR, 'H1.2b.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/H1.2b.json")

    if gain > 0.01:
        print(f"CONCLUSION: Pretrained MGM > Random MGM by {gain:.4f}. Hypothesis SUPPORTED.")
    elif abs(gain) <= 0.01:
        print("CONCLUSION: Pretrained ≈ Random. Pretraining adds no representation value at this scale.")
    else:
        print("CONCLUSION: Random > Pretrained. Unexpected — pretraining may be harmful.")

if __name__ == '__main__':
    main()
