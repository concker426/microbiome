#!/usr/bin/env python3
"""H1.1: Compare MGM encoder features vs raw genus features for IBD classification.

Trains 4 classifiers (RF, XGBoost, LogisticRegression, MLP) on:
  (a) 768-dim MGM encoder output
  (b) 86-dim raw genus IDs (ordered by abundance)
Reports 5-fold CV accuracy, F1, AUROC for each.
"""
import json, os, sys, time, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import torch
from run_v6_merged import MGMEnc  # reuse encoder definition

DATA = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
ENCODER_PATH = '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)

def load_data():
    train_data = []
    with open(os.path.join(DATA, 'train_nl.jsonl')) as f:
        for line in f:
            train_data.append(json.loads(line))
    test_data = []
    with open(os.path.join(DATA, 'test_nl.jsonl')) as f:
        for line in f:
            test_data.append(json.loads(line))

    train_seqs = np.load(os.path.join(DATA, 'train_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA, 'train_genus_masks.npy'))
    test_seqs = np.load(os.path.join(DATA, 'test_genus_sequences.npy'))
    test_masks = np.load(os.path.join(DATA, 'test_genus_masks.npy'))

    # Combine train+test for cross-validation
    all_data = train_data + test_data
    all_seqs = np.concatenate([train_seqs, test_seqs], axis=0)
    all_masks = np.concatenate([train_masks, test_masks], axis=0)
    all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in all_data])

    return all_data, all_seqs, all_masks, all_labels

def extract_mgm_features(seqs, masks, device='cuda:0'):
    """Extract 768-dim features from MGM encoder."""
    encoder = MGMEnc()
    ck = torch.load(ENCODER_PATH, map_location='cpu')
    st = ck.get('model_state_dict', ck)
    encoder.load_state_dict(st, strict=False)
    encoder.to(device)
    encoder.eval()

    features = []
    bs = 64
    for i in range(0, len(seqs), bs):
        batch_seqs = seqs[i:i+bs]
        batch_masks = masks[i:i+bs]
        gi = torch.from_numpy(batch_seqs.astype(np.int64)).long().to(device)
        gm = torch.from_numpy(batch_masks).bool().to(device)
        with torch.no_grad():
            feats = encoder(gi, gm)
        features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)

def evaluate_features(X, y, name, cv=5):
    """Run 4 classifiers with 5-fold CV on given features."""
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for clf_name, clf in classifiers.items():
        t0 = time.time()
        scores = cross_validate(clf, X, y, cv=skf, scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
        elapsed = time.time() - t0

        results[clf_name] = {
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'accuracy_std': float(np.std(scores['test_accuracy'])),
            'f1_mean': float(np.mean(scores['test_f1'])),
            'f1_std': float(np.std(scores['test_f1'])),
            'auroc_mean': float(np.mean(scores['test_roc_auc'])),
            'auroc_std': float(np.std(scores['test_roc_auc'])),
            'time_s': round(elapsed, 1),
        }
        print(f"  {clf_name:20s}: ACC={results[clf_name]['accuracy_mean']:.4f}±{results[clf_name]['accuracy_std']:.4f}  "
              f"F1={results[clf_name]['f1_mean']:.4f}  AUROC={results[clf_name]['auroc_mean']:.4f}")

    return results

def main():
    print("=" * 60)
    print("H1.1: MGM Encoder Features vs Raw Genus Features")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    all_data, all_seqs, all_masks, all_labels = load_data()
    n_samples = len(all_labels)
    n_pos = all_labels.sum()
    print(f"  Total: {n_samples} samples, Disease={n_pos} ({n_pos/n_samples*100:.1f}%), "
          f"Healthy={n_samples-n_pos} ({(n_samples-n_pos)/n_samples*100:.1f}%)")
    print(f"  Seqs shape: {all_seqs.shape}")

    print("\n[2/4] Extracting MGM encoder features (768-dim)...")
    t0 = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mgm_features = extract_mgm_features(all_seqs, all_masks, device)
    print(f"  MGM features shape: {mgm_features.shape}  ({time.time()-t0:.1f}s)")

    print("\n[3/4] Evaluating MGM features (768-dim)...")
    mgm_results = evaluate_features(mgm_features, all_labels, "MGM-768")

    print("\n[4/4] Evaluating Raw genus features (86-dim)...")
    # Raw: use genus IDs directly as integer features
    raw_features = all_seqs.astype(np.float32)
    # Normalize to [0,1] range
    raw_features = raw_features / 1226.0  # vocab size
    raw_results = evaluate_features(raw_features, all_labels, "Raw-86")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: MGM Encoder vs Raw Features (5-fold CV)")
    print("=" * 60)
    print(f"{'':20s} {'MGM-768':>20s} {'Raw-86':>20s} {'Δ':>10s}")
    print("-" * 72)
    for clf_name in ['RF', 'XGBoost', 'LogisticRegression', 'MLP']:
        mgm_acc = mgm_results[clf_name]['accuracy_mean']
        raw_acc = raw_results[clf_name]['accuracy_mean']
        delta = mgm_acc - raw_acc
        sign = '+' if delta > 0 else ''
        print(f"{clf_name:20s} {mgm_acc:>19.4f}  {raw_acc:>19.4f}  {sign}{delta:>9.4f}")

    # Best comparison
    best_mgm = max(mgm_results.items(), key=lambda x: x[1]['accuracy_mean'])
    best_raw = max(raw_results.items(), key=lambda x: x[1]['accuracy_mean'])
    print(f"\nBest MGM:  {best_mgm[0]} ACC={best_mgm[1]['accuracy_mean']:.4f} AUROC={best_mgm[1]['auroc_mean']:.4f}")
    print(f"Best Raw:  {best_raw[0]} ACC={best_raw[1]['accuracy_mean']:.4f} AUROC={best_raw[1]['auroc_mean']:.4f}")

    # Save structured results
    output = {
        'experiment': 'H1.1',
        'hypothesis': 'MGM encoder features >= raw genus features for classification',
        'n_samples': n_samples,
        'n_features_mgm': mgm_features.shape[1],
        'n_features_raw': raw_features.shape[1],
        'mgm_results': mgm_results,
        'raw_results': raw_results,
        'best_mgm': {'classifier': best_mgm[0], 'accuracy': best_mgm[1]['accuracy_mean'], 'auroc': best_mgm[1]['auroc_mean']},
        'best_raw': {'classifier': best_raw[0], 'accuracy': best_raw[1]['accuracy_mean'], 'auroc': best_raw[1]['auroc_mean']},
        'timestamp': str(__import__('datetime').datetime.now()),
        'metrics': {
            'best_mgm_accuracy': best_mgm[1]['accuracy_mean'],
            'best_raw_accuracy': best_raw[1]['accuracy_mean'],
            'mgm_advantage': best_mgm[1]['accuracy_mean'] - best_raw[1]['accuracy_mean'],
        }
    }

    with open(os.path.join(RESULT_DIR, 'H1.1.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_DIR}/H1.1.json")

    # Verdict
    if output['metrics']['mgm_advantage'] > 0.01:
        print("CONCLUSION: MGM encoder features OUTPERFORM raw features. Hypothesis SUPPORTED.")
    elif output['metrics']['mgm_advantage'] > -0.01:
        print("CONCLUSION: MGM encoder features are COMPARABLE to raw features.")
    else:
        print("CONCLUSION: Raw features OUTPERFORM MGM encoder. Hypothesis REJECTED.")

if __name__ == '__main__':
    main()
