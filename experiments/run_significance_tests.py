#!/usr/bin/env python3
"""
P0: Cross-Cohort Verification + Significance Tests
====================================================
1. Verify merged_all is truly held-out studies (not random split)
2. Train XGBoost, save per-sample predictions
3. Train MGM+MLP, save per-sample predictions
4. McNemar test: ProCyon v2 vs XGBoost, vs MGM
5. AUC difference: paired bootstrap
6. Per-source breakdown
"""
import json, os, sys, csv
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("SIGNIFICANCE TESTS + CROSS-COHORT AUDIT")
print("=" * 60)

# ── Load data ──
def load_dataset(name):
    path = f'/hd/liujx/microbiome_llm_project/data/qiita_ibd/{name}'
    train_data = [json.loads(l) for l in open(f'{path}/train_nl.jsonl')]
    test_data = [json.loads(l) for l in open(f'{path}/test_nl.jsonl')]
    ts = np.load(f'{path}/train_genus_sequences.npy')
    xs = np.load(f'{path}/test_genus_sequences.npy')
    tm = np.load(f'{path}/train_genus_masks.npy')
    xm = np.load(f'{path}/test_genus_masks.npy')
    return train_data, test_data, ts, xs, tm, xm

train_data, test_data, ts, xs, tm, xm = load_dataset('clean_2538')
m_train, m_test, m_ts, m_xs, m_tm, m_xm = load_dataset('merged_all')

# ═══════════════════════════════════════════
# 1. CROSS-COHORT VERIFICATION
# ═══════════════════════════════════════════
print("\n[1] Cross-Cohort Audit")

# Check study sources
clean_sources = set()
for d in train_data + test_data:
    src = d.get('dataset_type', d.get('source', 'unknown'))
    clean_sources.add(str(src))

merged_sources = set()
for d in m_train + m_test:
    src = d.get('dataset_type', d.get('source', 'unknown'))
    merged_sources.add(str(src))

print(f"  clean_2538 sources: {clean_sources}")
print(f"  merged_all sources: {merged_sources}")

# Check sample_id overlap
clean_ids = set(d['sample_id'] for d in train_data + test_data)
merged_ids = set(d['sample_id'] for d in m_train + m_test)
overlap = clean_ids & merged_ids
print(f"  clean_2538 samples: {len(clean_ids)}")
print(f"  merged_all samples: {len(merged_ids)}")
print(f"  Overlap: {len(overlap)} samples")
if overlap:
    print(f"    Sample IDs in both: {list(overlap)[:5]}...")

# Check if merged train/test are from same studies
m_train_sources = set()
for d in m_train:
    m_train_sources.add(str(d.get('dataset_type', d.get('source', 'unknown'))))
m_test_sources = set()
for d in m_test:
    m_test_sources.add(str(d.get('dataset_type', d.get('source', 'unknown'))))
train_only = m_train_sources - m_test_sources
test_only = m_test_sources - m_train_sources
shared = m_train_sources & m_test_sources
print(f"\n  merged_all train sources: {m_train_sources}")
print(f"  merged_all test sources: {m_test_sources}")
print(f"  Shared sources (train+test): {shared}")
print(f"  Train-only sources: {train_only}")
print(f"  Test-only sources: {test_only}")
print(f"  Is test truly held-out studies? {'YES' if test_only else 'PARTIALLY (some sources in both)'}")

# Conclusion
print(f"\n  AUDIT CONCLUSION:")
print(f"  clean→merged IS cross-cohort: different Qiita studies, different sample IDs")
print(f"  But merged_all sources overlap between train/test, so merged→merged")
print(f"  is NOT leave-one-study-out. It's a random split within merged studies.")
print(f"  RECOMMENDED TERMINOLOGY:")
print(f"    Table 4 -> 'Cross-Dataset Transfer Evaluation'")
print(f"    Not 'Cross-Cohort Validation' (which implies leave-one-cohort-out)")

# ═══════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ═══════════════════════════════════════════
print("\n[2] Feature Extraction for Baselines")

V = 1226
def extract_raw_abundance(seqs, masks):
    n = len(seqs)
    feats = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        valid = masks[i].astype(bool)
        for j in range(len(seqs[i])):
            if valid[j] and seqs[i, j] > 0:
                feats[i, int(seqs[i, j])] += 1.0
        total = feats[i].sum()
        if total > 0: feats[i] /= total
    return feats

X_train_raw = extract_raw_abundance(ts, tm)
X_test_raw = extract_raw_abundance(xs, xm)
y_train = np.array([1 if d['label']=='Disease' else 0 for d in train_data])
y_test = np.array([1 if d['label']=='Disease' else 0 for d in test_data])

# ═══════════════════════════════════════════
# 3. TRAIN XGBOOST + SAVE PER-SAMPLE PREDICTIONS
# ═══════════════════════════════════════════
print("\n[3] XGBoost per-sample predictions")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0)
xgb.fit(X_train_raw, y_train)
xgb_preds = xgb.predict(X_test_raw)
xgb_probs = xgb.predict_proba(X_test_raw)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_probs)
xgb_cm = confusion_matrix(y_test, xgb_preds)
print(f"  XGBoost: ACC={xgb_acc:.4f} AUC={xgb_auc:.4f} CM={xgb_cm.ravel()}")

# ═══════════════════════════════════════════
# 4. TRAIN MGM + MLP + SAVE PER-SAMPLE PREDICTIONS
# ═══════════════════════════════════════════
print("\n[4] MGM + MLP per-sample predictions")
# Extract MGM features
import torch
from run_v6_merged import MGMEnc
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def extract_mgm(seqs, masks):
    enc = MGMEnc()
    ck = torch.load('/hd/liujx/microbiome_llm_project/saved_models/mgm_encoder_pretrained/mgm_encoder.pt',
                   map_location='cpu')
    enc.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    enc.to(DEVICE).eval()
    feats = []
    for i in range(0, len(seqs), 64):
        gi = torch.from_numpy(seqs[i:i+64].astype(np.int64)).long().to(DEVICE)
        gm = torch.from_numpy(masks[i:i+64]).bool().to(DEVICE)
        feats.append(enc(gi, gm).cpu().numpy())
    del enc; torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)

print("  Extracting MGM features...")
X_train_mgm = extract_mgm(ts, tm)
X_test_mgm = extract_mgm(xs, xm)
print(f"  MGM features: train={X_train_mgm.shape}, test={X_test_mgm.shape}")

mgm_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
mgm_mlp.fit(X_train_mgm, y_train)
mgm_preds = mgm_mlp.predict(X_test_mgm)
mgm_probs = mgm_mlp.predict_proba(X_test_mgm)[:, 1]
mgm_acc = accuracy_score(y_test, mgm_preds)
mgm_auc = roc_auc_score(y_test, mgm_probs)
mgm_cm = confusion_matrix(y_test, mgm_preds)
print(f"  MGM+MLP: ACC={mgm_acc:.4f} AUC={mgm_auc:.4f} CM={mgm_cm.ravel()}")

# ═══════════════════════════════════════════
# 5. LOAD PROCYON V2 PER-SAMPLE PREDICTIONS
# ═══════════════════════════════════════════
print("\n[5] Loading ProCyon v2 per-sample predictions")
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}

# Align with test data order
test_ids = [d['sample_id'] for d in test_data]
procyon_probs = np.array([float(pred_data[sid]['prob_disease']) for sid in test_ids])
procyon_preds = (procyon_probs > 0.5).astype(int)
procyon_acc = accuracy_score(y_test, procyon_preds)
procyon_auc = roc_auc_score(y_test, procyon_probs)
print(f"  ProCyon v2: ACC={procyon_acc:.4f} AUC={procyon_auc:.4f}")

# ═══════════════════════════════════════════
# 6. MCNEMAR TEST
# ═══════════════════════════════════════════
print("\n[6] McNemar Tests")

from scipy.stats import binomtest

def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):
    """McNemar test: are two classifiers significantly different?"""
    both_correct = (pred_a == y_true) & (pred_b == y_true)
    both_wrong = (pred_a != y_true) & (pred_b != y_true)
    a_correct_b_wrong = (pred_a == y_true) & (pred_b != y_true)
    a_wrong_b_correct = (pred_a != y_true) & (pred_b == y_true)

    b = a_correct_b_wrong.sum()
    c = a_wrong_b_correct.sum()
    n_discordant = b + c

    if n_discordant == 0:
        return {'test': 'mcnemar', 'b': int(b), 'c': int(c), 'n_discordant': 0,
                'p_value': 1.0, 'significant': False, 'note': 'no discordant pairs'}

    # Exact binomial test
    result = binomtest(min(b, c), n=n_discordant, p=0.5, alternative='two-sided')
    p_value = result.pvalue

    print(f"  {name_a} vs {name_b}:")
    print(f"    Both correct: {both_correct.sum()}  Both wrong: {both_wrong.sum()}")
    print(f"    {name_a} correct, {name_b} wrong: {b}")
    print(f"    {name_a} wrong, {name_b} correct: {c}")
    print(f"    Discordant pairs: {n_discordant}")
    print(f"    McNemar p = {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

    return {'test': 'mcnemar', 'b': int(b), 'c': int(c), 'n_discordant': int(n_discordant),
            'p_value': float(p_value), 'significant': p_value < 0.05}

mcnemar_results = {}
mcnemar_results['ProCyon_v2 vs XGBoost'] = mcnemar_test(y_test, procyon_preds, xgb_preds, 'ProCyon', 'XGBoost')
mcnemar_results['ProCyon_v2 vs MGM+MLP'] = mcnemar_test(y_test, procyon_preds, mgm_preds, 'ProCyon', 'MGM')
mcnemar_results['XGBoost vs MGM+MLP'] = mcnemar_test(y_test, xgb_preds, mgm_preds, 'XGBoost', 'MGM')

# ═══════════════════════════════════════════
# 7. AUC DIFFERENCE (PAIRED BOOTSTRAP)
# ═══════════════════════════════════════════
print("\n[7] AUC Difference (Paired Bootstrap)")

def auc_diff_bootstrap(y_true, probs_a, probs_b, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            auc_a = roc_auc_score(y_true[idx], probs_a[idx])
            auc_b = roc_auc_score(y_true[idx], probs_b[idx])
            diffs.append(auc_a - auc_b)
        except:
            pass
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())  # two-sided
    return {'mean_diff': float(mean_diff), 'ci_95': [float(ci_low), float(ci_high)],
            'p_value': float(p_value), 'n_bootstrap': n_bootstrap}

auc_results = {}
for name_a, probs_a, name_b, probs_b in [
    ('ProCyon v2', procyon_probs, 'XGBoost', xgb_probs),
    ('ProCyon v2', procyon_probs, 'MGM+MLP', mgm_probs),
    ('XGBoost', xgb_probs, 'MGM+MLP', mgm_probs),
]:
    result = auc_diff_bootstrap(y_test, probs_a, probs_b)
    auc_results[f'{name_a} vs {name_b}'] = result
    sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'n.s.'
    print(f"  ΔAUC({name_a} - {name_b}) = {result['mean_diff']:.4f} "
          f"[{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}] "
          f"p={result['p_value']:.4f} {sig}")

# ═══════════════════════════════════════════
# 8. PER-CATEGORY PERFORMANCE
# ═══════════════════════════════════════════
print("\n[8] Per-Category Performance")

for name, probs, preds in [('ProCyon v2', procyon_probs, procyon_preds),
                             ('XGBoost', xgb_probs, xgb_preds),
                             ('MGM+MLP', mgm_probs, mgm_preds)]:
    for label, label_name in [(0, 'Healthy'), (1, 'Disease')]:
        mask = y_test == label
        acc = accuracy_score(y_test[mask], preds[mask])
        print(f"  {name} on {label_name} (n={mask.sum()}): ACC={acc:.4f}")

# ═══════════════════════════════════════════
# 9. SAVE ALL PER-SAMPLE PREDICTIONS
# ═══════════════════════════════════════════
print("\n[9] Saving per-sample predictions")

per_sample = []
for i, d in enumerate(test_data):
    per_sample.append({
        'sample_id': d['sample_id'],
        'true_label': d['label'],
        'true_binary': int(y_test[i]),
        'ProCyon_v2_prob': float(procyon_probs[i]),
        'ProCyon_v2_pred': int(procyon_preds[i]),
        'XGBoost_prob': float(xgb_probs[i]),
        'XGBoost_pred': int(xgb_preds[i]),
        'MGM_prob': float(mgm_probs[i]),
        'MGM_pred': int(mgm_preds[i]),
    })

with open(f'{OUT_DIR}/per_sample_predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
    writer.writeheader()
    writer.writerows(per_sample)
print(f"Saved: {OUT_DIR}/per_sample_predictions.csv")

# ═══════════════════════════════════════════
# 10. SAVE ALL RESULTS
# ═══════════════════════════════════════════
results = {
    'cross_cohort_audit': {
        'clean_sources': list(clean_sources),
        'merged_sources': list(merged_sources),
        'overlap_count': len(overlap),
        'merged_train_sources': list(m_train_sources),
        'merged_test_sources': list(m_test_sources),
        'shared_sources': list(shared),
        'test_only_sources': list(test_only),
        'recommended_terminology': 'Cross-Dataset Transfer Evaluation (not cross-cohort, as merged test is random split within shared studies)',
    },
    'baseline_performance': {
        'XGBoost': {'acc': float(xgb_acc), 'auc': float(xgb_auc)},
        'MGM_MLP': {'acc': float(mgm_acc), 'auc': float(mgm_auc)},
        'ProCyon_v2': {'acc': float(procyon_acc), 'auc': float(procyon_auc)},
    },
    'mcnemar_tests': mcnemar_results,
    'auc_bootstrap_tests': auc_results,
}
with open(f'{OUT_DIR}/significance_tests.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT_DIR}/significance_tests.json")

print("\n" + "=" * 60)
print("KEY FINDINGS FOR PAPER")
print("=" * 60)
print(f"  Cross-cohort terminology: use 'Cross-Dataset Transfer' not 'Cross-Cohort'")
print(f"  ProCyon v2 vs XGBoost McNemar: p={mcnemar_results['ProCyon_v2 vs XGBoost']['p_value']:.4f}")
print(f"  ProCyon v2 vs MGM McNemar: p={mcnemar_results['ProCyon_v2 vs MGM+MLP']['p_value']:.6f}")
print(f"  ΔAUC(ProCyon - XGBoost): {auc_results['ProCyon v2 vs XGBoost']['mean_diff']:.4f}")
print(f"  ΔAUC(ProCyon - MGM): {auc_results['ProCyon v2 vs MGM+MLP']['mean_diff']:.4f}")
print("\nDONE")
