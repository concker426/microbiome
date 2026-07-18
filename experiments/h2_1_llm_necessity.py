#!/usr/bin/env python3
"""H2.1: Is the LLM necessary for classification?

Builds a comparison table of:
  - Encoder+MLP:  MGM features -> MLP classifier          (from H1.1)
  - Encoder+LLM:  MGM features -> LLM text generation     (from V5/V6 results)
  - LLM NL-only:  Pure LLM with NL text input             (from V6 dropout eval)
  - Raw+MLP:      Raw genus features -> MLP classifier    (from H1.1)
  - Pure LLM:     No encoder, no genus, just NL text       (from H1.2)

Collects from existing results, fills gaps by running what's missing.
"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import torch
from run_v6_merged import MGMEnc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
ENCODER_PATH = '/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt'
RESULT_DIR = '/hd/liujx/microbiome_llm_project/experiments/results'
os.makedirs(RESULT_DIR, exist_ok=True)

def load_clean2538():
    data = []
    with open(os.path.join(DATA_DIR, 'train_nl.jsonl')) as f:
        for line in f: data.append(json.loads(line))
    test_data = []
    with open(os.path.join(DATA_DIR, 'test_nl.jsonl')) as f:
        for line in f: test_data.append(json.loads(line))
    all_data = data + test_data
    train_seqs = np.load(os.path.join(DATA_DIR, 'train_genus_sequences.npy'))
    test_seqs = np.load(os.path.join(DATA_DIR, 'test_genus_sequences.npy'))
    train_masks = np.load(os.path.join(DATA_DIR, 'train_genus_masks.npy'))
    test_masks = np.load(os.path.join(DATA_DIR, 'test_genus_masks.npy'))
    all_seqs = np.concatenate([train_seqs, test_seqs], axis=0)
    all_masks = np.concatenate([train_masks, test_masks], axis=0)
    all_labels = np.array([1 if d['label'] == 'Disease' else 0 for d in all_data])
    return all_seqs, all_masks, all_labels

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
        with torch.no_grad(): feats.append(enc(gi, gm).cpu().numpy())
    return np.concatenate(feats, axis=0)

def evaluate_mlp(X, y, name, n_repeats=5):
    """MLP with stratified 5-fold CV, repeated."""
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for _ in range(n_repeats):
        cv_scores = cross_val_score(mlp, X, y, cv=skf, scoring='accuracy')
        scores.extend(cv_scores)
    return np.mean(scores), np.std(scores)

def load_previous_results():
    """Try to load H1.1 and H1.2 results."""
    results = {}
    for exp_id in ['H1.1', 'H1.2']:
        path = os.path.join(RESULT_DIR, f'{exp_id}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[exp_id] = json.load(f)
    return results

def load_v5_v6_results():
    """Load known V5/V6 benchmark results."""
    # These are hardcoded from completed experiments
    return {
        'V5_baseline': {
            'enc_nl_acc': 0.8862, 'nl_only_acc': 0.5629,
            'source': 'V5 weighted, NMT=4, PS=0.1, clean_2538'
        },
        'V6b_curriculum': {
            'enc_nl_acc': 0.8623, 'nl_only_acc': 0.7605,
            'source': 'V6b curriculum 30->80%, NMT=4, PS=0.1, clean_2538'
        },
        'V6_modality_dropout': {
            'enc_nl_acc': 0.8503, 'nl_only_acc': 0.6946,
            'source': 'V6 fixed 50% dropout, NMT=4, PS=0.1, clean_2538'
        },
    }

def main():
    print("=" * 60)
    print("H2.1: LLM Necessity for Classification")
    print("=" * 60)

    prev = load_previous_results()
    bench = load_v5_v6_results()

    # Build comparison rows
    rows = {}

    # 1. Raw features + best ML (from H1.1 or compute now)
    print("\n[1/5] Raw genus features + MLP...")
    if 'H1.1' in prev and 'raw_results' in prev['H1.1']:
        best_raw = max(prev['H1.1']['raw_results'].items(), key=lambda x: x[1]['accuracy_mean'])
        rows['Raw + Best ML'] = {
            'acc': best_raw[1]['accuracy_mean'],
            'acc_std': best_raw[1]['accuracy_std'],
            'model': best_raw[0],
            'description': 'Raw 86-dim genus IDs -> best sklearn classifier',
            'category': 'ML baseline'
        }
    else:
        all_seqs, all_masks, all_labels = load_clean2538()
        raw_feats = all_seqs.astype(np.float32) / 1226.0
        acc, std = evaluate_mlp(raw_feats, all_labels, 'Raw+MLP')
        rows['Raw + MLP'] = {'acc': acc, 'acc_std': std, 'model': 'MLP(256,128)', 'description': '86-dim raw features -> MLP', 'category': 'ML baseline'}

    # 2. MGM features + MLP (from H1.1 or compute now)
    print("[2/5] MGM encoder features + MLP...")
    if 'H1.1' in prev and 'mgm_results' in prev['H1.1']:
        best_mgm = max(prev['H1.1']['mgm_results'].items(), key=lambda x: x[1]['accuracy_mean'])
        rows['Encoder + MLP'] = {
            'acc': best_mgm[1]['accuracy_mean'],
            'acc_std': best_mgm[1]['accuracy_std'],
            'model': best_mgm[0],
            'description': 'MGM 768-dim features -> best sklearn classifier',
            'category': 'Encoder + MLP'
        }
    else:
        all_seqs, all_masks, all_labels = load_clean2538()
        mgm_feats = extract_mgm_features(all_seqs, all_masks)
        acc, std = evaluate_mlp(mgm_feats, all_labels, 'MGM+MLP')
        rows['Encoder + MLP'] = {'acc': acc, 'acc_std': std, 'model': 'MLP(256,128)', 'description': 'MGM 768-dim -> MLP', 'category': 'Encoder + MLP'}

    # 3. Encoder + LLM (from V5)
    print("[3/5] Encoder + LLM (V5 baseline)...")
    rows['Encoder + LLM (V5)'] = {
        'acc': bench['V5_baseline']['enc_nl_acc'],
        'category': 'Encoder + LLM',
        'description': 'MGM -> 4 projection tokens -> Qwen2.5-7B LoRA -> NL diagnosis',
        'model': 'ProCyon V5'
    }

    # 4. Encoder + LLM (V6b, best NL-only)
    print("[4/5] Encoder + LLM (V6b curriculum dropout)...")
    rows['Encoder + LLM (V6b)'] = {
        'acc': bench['V6b_curriculum']['enc_nl_acc'],
        'category': 'Encoder + LLM',
        'description': 'MGM -> projection + curriculum dropout -> Qwen2.5-7B LoRA',
        'model': 'ProCyon V6b'
    }

    # 5. LLM NL-only (V6b dropout path = no encoder at test time)
    rows['LLM NL-only (V6b)'] = {
        'acc': bench['V6b_curriculum']['nl_only_acc'],
        'category': 'LLM text-only',
        'description': 'Zero encoder + Qwen2.5-7B LoRA, trained with dropout',
        'model': 'ProCyon V6b (dropout mode)'
    }

    # 6. Pure LLM no training (from H1.2 if available)
    print("[5/5] Pure LLM (no encoder, no training)...")
    if 'H1.2' in prev and 'results' in prev['H1.2']:
        no_enc = prev['H1.2']['results'].get('no_encoder', {})
        if 'accuracy' in no_enc:
            rows['Pure LLM (untrained)'] = {
                'acc': no_enc['accuracy'],
                'category': 'LLM text-only',
                'description': 'Qwen2.5-7B-Instruct zero-shot NL diagnosis',
                'model': 'Qwen2.5-7B-Instruct (no FT)'
            }

    # Sort rows by accuracy
    sorted_rows = sorted(rows.items(), key=lambda x: x[1]['acc'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 80)
    print("H2.1: Classification Performance Comparison")
    print("=" * 80)
    print(f"{'Method':<30} {'Category':<20} {'Accuracy':>10} {'±Std':>8}")
    print("-" * 80)
    for name, row in sorted_rows:
        std = row.get('acc_std', 0)
        print(f"{name:<30} {row['category']:<20} {row['acc']:>10.4f} {std:>8.4f}")

    # Key comparisons
    print("\n--- Key Comparisons ---")
    if 'Encoder + MLP' in rows and 'Encoder + LLM (V5)' in rows:
        mlp_acc = rows['Encoder + MLP']['acc']
        llm_acc = rows['Encoder + LLM (V5)']['acc']
        diff = mlp_acc - llm_acc
        print(f"  MLP vs LLM (same encoder): MLP {mlp_acc:.4f}, LLM {llm_acc:.4f}, Δ={diff:+.4f}")
        if abs(diff) < 0.03:
            print("  → MLP and LLM are COMPARABLE for classification. LLM adds NL explanation.")

    if 'Encoder + MLP' in rows and 'Raw + Best ML' in rows:
        enc_acc = rows['Encoder + MLP']['acc']
        raw_acc = rows['Raw + Best ML']['acc']
        diff = enc_acc - raw_acc
        print(f"  MGM vs Raw features: MGM {enc_acc:.4f}, Raw {raw_acc:.4f}, Δ={diff:+.4f}")

    if 'LLM NL-only (V6b)' in rows and 'Encoder + LLM (V6b)' in rows:
        nl_acc = rows['LLM NL-only (V6b)']['acc']
        enc_acc = rows['Encoder + LLM (V6b)']['acc']
        gap = enc_acc - nl_acc
        print(f"  Enc+NL vs NL-only gap: {gap:.4f} (modality collapse residual)")

    # Save
    output = {
        'experiment': 'H2.1',
        'hypothesis': 'LLM is NOT necessary for classification, but provides NL explanation and multi-task capability',
        'comparison_table': {name: row for name, row in sorted_rows},
        'key_findings': {
            'mlp_vs_llm_gap': abs(rows.get('Encoder + MLP', {}).get('acc', 0) - rows.get('Encoder + LLM (V5)', {}).get('acc', 0)),
            'encoder_advantage': rows.get('Encoder + MLP', {}).get('acc', 0) - rows.get('Raw + MLP', {}).get('acc', 0) if 'Raw + MLP' in rows else rows.get('Encoder + MLP', {}).get('acc', 0) - rows.get('Raw + Best ML', {}).get('acc', 0),
            'encoder_nl_gap': rows.get('Encoder + LLM (V6b)', {}).get('acc', 0) - rows.get('LLM NL-only (V6b)', {}).get('acc', 0),
        },
        'timestamp': str(__import__('datetime').datetime.now()),
        'metrics': {
            'best_overall_acc': sorted_rows[0][1]['acc'],
            'best_overall_method': sorted_rows[0][0],
        }
    }

    with open(os.path.join(RESULT_DIR, 'H2.1.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_DIR}/H2.1.json")

    # Verdict
    print("\nCONCLUSION:")
    if 'Encoder + MLP' in rows and abs(rows['Encoder + MLP']['acc'] - rows.get('Encoder + LLM (V5)', {}).get('acc', 0)) < 0.03:
        print("  - MLP ~= LLM for CLASSIFICATION: encoder quality matters more than LLM")
    print("  - LLM value is in NL explanation, multi-task, and interactive QA, not pure classification")
    print("  - Modality dropout is ESSENTIAL for LLM to learn from NL text")

if __name__ == '__main__':
    main()
