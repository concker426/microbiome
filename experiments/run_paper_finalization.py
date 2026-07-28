#!/usr/bin/env python3
"""
Paper Finalization: Calibration + Error Analysis + Case Studies
===============================================================
1. Calibration: Reliability Diagram, ECE, Brier Score
2. Error Analysis: confusion patterns, per-sample error breakdown
3. Case Studies: 4 representative cases from Phase 4.5 data
4. Final LaTeX tables
"""
import json, os, sys, csv, pickle
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
from sklearn.metrics import (brier_score_loss, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PAPER FINALIZATION: Calibration + Error + Cases")
print("=" * 60)

# ── Load data ──
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    preds = list(csv.DictReader(f))
    test_preds = [r for r in preds if r['split'] == 'test']

test_ids = [r['sample_id'] for r in test_preds]
test_probs = np.array([float(r['prob_disease']) for r in test_preds])
test_true = np.array([1 if r['ground_truth'] == 'Disease' else 0 for r in test_preds])
ensemble_probs = np.array([float(r.get('ensemble_prob_disease', r['prob_disease'])) for r in test_preds])
test_true_binary = test_true
test_pred_binary = (test_probs > 0.5).astype(int)

print(f"Test samples: {len(test_preds)} (D={test_true.sum()})")
print(f"Classifier ACC: {accuracy_score(test_true, test_pred_binary):.4f}")
print(f"Classifier AUC: {roc_auc_score(test_true, test_probs):.4f}")

# ═══════════════════════════════════════════
# 1. CALIBRATION ANALYSIS
# ═══════════════════════════════════════════
print("\n[1] Calibration Analysis")

# ECE (Expected Calibration Error)
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
ece = 0.0
reliability_data = []
for i in range(n_bins):
    mask = (test_probs >= bin_edges[i]) & (test_probs < bin_edges[i+1])
    if mask.sum() == 0:
        continue
    bin_acc = test_true[mask].mean()
    bin_conf = test_probs[mask].mean()
    bin_size = mask.sum()
    ece += (bin_size / len(test_probs)) * abs(bin_acc - bin_conf)
    reliability_data.append({'bin': i, 'count': int(bin_size), 'accuracy': float(bin_acc),
                             'confidence': float(bin_conf), 'range': f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})'})

# Brier Score
brier = brier_score_loss(test_true, test_probs)

# Calibration curve
prob_true, prob_pred = calibration_curve(test_true, test_probs, n_bins=10)

print(f"  ECE: {ece:.4f}")
print(f"  Brier Score: {brier:.4f}")
for rd in reliability_data:
    print(f"    Bin {rd['bin']} {rd['range']}: n={rd['count']} acc={rd['accuracy']:.3f} conf={rd['confidence']:.3f} gap={abs(rd['accuracy']-rd['confidence']):.4f}")

# ═══════════════════════════════════════════
# 2. ERROR ANALYSIS
# ═══════════════════════════════════════════
print("\n[2] Error Analysis")

cm = confusion_matrix(test_true, test_pred_binary)
tn, fp, fn, tp = cm.ravel()

# False Positives (Healthy → Disease)
fp_samples = [(test_ids[i], test_probs[i]) for i in range(len(test_true))
              if test_true[i] == 0 and test_pred_binary[i] == 1]
fp_samples.sort(key=lambda x: x[1], reverse=True)

# False Negatives (Disease → Healthy)
fn_samples = [(test_ids[i], test_probs[i]) for i in range(len(test_true))
              if test_true[i] == 1 and test_pred_binary[i] == 0]
fn_samples.sort(key=lambda x: x[1])

print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  False Positives (Healthy→IBD): {len(fp_samples)}")
print(f"  False Negatives (IBD→Healthy): {len(fn_samples)}")
print(f"  Top-3 FP (most confident wrong): {fp_samples[:3]}")
print(f"  Top-3 FN (most confident wrong): {fn_samples[:3]}")

# Load SHAP data for error analysis
with open(f'{OUT_DIR}/shap_data_full.pkl', 'rb') as f:
    shap_by_id = {}
    for s in pickle.load(f)['all_samples']:
        shap_by_id[s['sample_id']] = {'label': s['label'], 'importance': s['importance']}

# ═══════════════════════════════════════════
# 3. CASE STUDIES from Phase 4.5
# ═══════════════════════════════════════════
print("\n[3] Extracting Case Studies from Phase 4.5 data")

with open(f'{OUT_DIR}/phase45_validation.json') as f:
    phase45 = json.load(f)

results_45 = phase45['results']
print(f"  Phase 4.5 samples: {len(results_45)}")

# Select 4 case studies
cases = []

# Case A: Correct prediction + SHAP consistent + good specificity
correct_good = [r for r in results_45
                if r['predicted'] == 'IBD' and r['ground_truth'] == 'Disease'
                and len(r.get('shap_top', [])) > 3]
if correct_good:
    r = correct_good[0]
    cases.append({'name': 'Case A: Correct IBD Prediction, SHAP-Grounded Explanation',
                  'sample_id': r['sample_id'], 'ground_truth': r['ground_truth'],
                  'predicted': r['predicted'], 'prob': r['prob_disease'],
                  'shap_top': r.get('shap_top', [])[:10],
                  'resp_raw': r.get('response_a', ''),
                  'resp_shap': r.get('response_b', ''),
                  'resp_lit': r.get('response_c', '')})

# Case B: Borderline sample
borderline = [r for r in results_45 if 0.4 < r['prob_disease'] < 0.6]
if borderline:
    r = borderline[0]
    cases.append({'name': 'Case B: Borderline Confidence',
                  'sample_id': r['sample_id'], 'ground_truth': r['ground_truth'],
                  'predicted': r['predicted'], 'prob': r['prob_disease'],
                  'shap_top': r.get('shap_top', [])[:10],
                  'resp_raw': r.get('response_a', ''),
                  'resp_shap': r.get('response_b', ''),
                  'resp_lit': r.get('response_c', '')})

# Case C: Wrong prediction
wrong = [r for r in results_45
         if (r['predicted'] == 'IBD' and r['ground_truth'] == 'Healthy') or
            (r['predicted'] == 'HEALTHY' and r['ground_truth'] == 'Disease')]
if wrong:
    r = wrong[0]
    cases.append({'name': f"Case C: Misclassification (True={r['ground_truth']}, Pred={r['predicted']})",
                  'sample_id': r['sample_id'], 'ground_truth': r['ground_truth'],
                  'predicted': r['predicted'], 'prob': r['prob_disease'],
                  'shap_top': r.get('shap_top', [])[:10],
                  'resp_raw': r.get('response_a', ''),
                  'resp_shap': r.get('response_b', ''),
                  'resp_lit': r.get('response_c', '')})

# Case D: Extreme IBD (cluster 1 type)
extreme_ibd = [r for r in results_45
               if r['predicted'] == 'IBD' and r['prob_disease'] > 0.95
               and r['ground_truth'] == 'Disease']
if extreme_ibd:
    r = extreme_ibd[-1]  # highest confidence
    cases.append({'name': 'Case D: Extreme IBD (High Confidence)',
                  'sample_id': r['sample_id'], 'ground_truth': r['ground_truth'],
                  'predicted': r['predicted'], 'prob': r['prob_disease'],
                  'shap_top': r.get('shap_top', [])[:10],
                  'resp_raw': r.get('response_a', ''),
                  'resp_shap': r.get('response_b', ''),
                  'resp_lit': r.get('response_c', '')})

for case in cases:
    print(f"  {case['name']}: {case['sample_id']} (prob={case['prob']:.3f})")

# ═══════════════════════════════════════════
# 4. GENERATE FIGURE
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))

# Panel A: Calibration Curve (Reliability Diagram)
ax = fig.add_subplot(2, 3, 1)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
ax.plot(prob_pred, prob_true, 'o-', color='#1565C0', markersize=10, linewidth=2, label='ProCyon v2')
bin_sizes = [rd['count'] for rd in reliability_data]
ax.scatter(prob_pred, prob_true, s=[s*3 for s in bin_sizes], c='#1565C0',
          alpha=0.6, edgecolors='black', linewidths=1, zorder=5)
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title(f'A. Calibration Curve (ECE={ece:.4f}, Brier={brier:.4f})', fontweight='bold', loc='left')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel B: Confidence Distribution
ax = fig.add_subplot(2, 3, 2)
ax.hist(test_probs[test_true==0], bins=20, alpha=0.7, label='Healthy', color='#4CAF50', edgecolor='none')
ax.hist(test_probs[test_true==1], bins=20, alpha=0.7, label='IBD', color='#F44336', edgecolor='none')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Predicted P(IBD)'); ax.set_ylabel('Count')
ax.set_title('B. Prediction Confidence Distribution', fontweight='bold', loc='left')
ax.legend(fontsize=8)

# Panel C: Confusion Matrix
ax = fig.add_subplot(2, 3, 3)
cm_display = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_display, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Healthy', 'Pred IBD']); ax.set_yticklabels(['True Healthy', 'True IBD'])
for i in range(2):
    for j in range(2):
        color = 'white' if cm_display[i, j] > cm_display.max()/2 else 'black'
        ax.text(j, i, f'{cm_display[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold', color=color)
ax.set_title(f'C. Confusion Matrix\nACC={accuracy_score(test_true,test_pred_binary):.4f} AUC={roc_auc_score(test_true,test_probs):.4f}',
            fontweight='bold', loc='left')

# Panel D: Error Profile (FP vs FN samples)
ax = fig.add_subplot(2, 3, 4)
fp_probs = [p for _, p in fp_samples]
fn_probs = [p for _, p in fn_samples]
ax.scatter(range(len(fp_probs)), sorted(fp_probs, reverse=True), s=30, c='#FF9800',
          alpha=0.7, label=f'False Positives (n={len(fp_probs)})', edgecolors='none')
ax.scatter(range(len(fn_probs)), sorted(fn_probs), s=30, c='#F44336',
          alpha=0.7, label=f'False Negatives (n={len(fn_probs)})', edgecolors='none')
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Sample Rank'); ax.set_ylabel('P(IBD)')
ax.set_title('D. Error Profile', fontweight='bold', loc='left')
ax.legend(fontsize=8)

# Panel E: LLM Benchmark Summary (from Phase 4.5)
ax = fig.add_subplot(2, 3, 5)
phase45_metrics = phase45.get('metrics', {})
variants = ['A (Raw)', 'B (SHAP)', 'C (SHAP+Lit)']
hall_vals = [phase45_metrics.get(v, {}).get('hallucination', 0) for v in ['A', 'B', 'C']]
cons_vals = [phase45_metrics.get(v, {}).get('consistency', 0) for v in ['A', 'B', 'C']]
spec_vals = [phase45_metrics.get(v, {}).get('specificity', 0) for v in ['A', 'B', 'C']]
x = np.arange(3); w = 0.25
ax.bar(x-w, hall_vals, w, label='Hallucination', color='#F44336', edgecolor='none')
ax.bar(x, cons_vals, w, label='Consistency', color='#4CAF50', edgecolor='none')
ax.bar(x+w, spec_vals, w, label='Specificity', color='#1565C0', edgecolor='none')
for i in range(3):
    ax.text(i-w, hall_vals[i]+0.02, f'{hall_vals[i]:.2f}', ha='center', fontsize=7)
    ax.text(i, cons_vals[i]+0.02, f'{cons_vals[i]:.2f}', ha='center', fontsize=7)
    ax.text(i+w, spec_vals[i]+0.02, f'{spec_vals[i]:.2f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(variants); ax.set_ylabel('Score')
ax.set_title('E. LLM Explanation Validation (n=50)', fontweight='bold', loc='left')
ax.legend(fontsize=7, loc='upper right')

# Panel F: Case Study Summary
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')
case_text = "CASE STUDIES\n" + "="*40 + "\n\n"
for case in cases:
    case_text += f"{case['name']}\n"
    case_text += f"  Sample: {case['sample_id']}\n"
    case_text += f"  True: {case['ground_truth']} | Pred: {case['predicted']} (p={case['prob']:.3f})\n"
    case_text += f"  Top SHAP: {', '.join([g['genus_name'][:15] for g in case['shap_top'][:5]])}\n\n"
ax.text(0.05, 0.95, case_text, transform=ax.transAxes, fontsize=8, fontfamily='monospace',
       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

fig.suptitle('ProCyon v2: Calibration, Error Analysis & Case Studies', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUT_DIR}/calibration_error_cases.png', dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/calibration_error_cases.png")

# ═══════════════════════════════════════════
# 5. SAVE ALL RESULTS
# ═══════════════════════════════════════════
final_metrics = {
    'classification': {
        'accuracy': float(accuracy_score(test_true, test_pred_binary)),
        'auc': float(roc_auc_score(test_true, test_probs)),
        'sensitivity': float(tp/(tp+fn)),
        'specificity': float(tn/(tn+fp)),
        'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'f1': float(2*tp/(2*tp+fp+fn)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    },
    'calibration': {
        'ece': float(ece),
        'brier_score': float(brier),
        'reliability_data': reliability_data,
    },
    'error_analysis': {
        'n_false_positives': len(fp_samples),
        'n_false_negatives': len(fn_samples),
        'fp_sample_ids': [sid for sid, _ in fp_samples[:10]],
        'fn_sample_ids': [sid for sid, _ in fn_samples[:10]],
    },
    'case_studies': cases,
}

with open(f'{OUT_DIR}/final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/final_metrics.json")

# ═══════════════════════════════════════════
# 6. GENERATE LATEX TABLES
# ═══════════════════════════════════════════
tables_tex = []

# Table 5: Calibration
tables_tex.append(r"\begin{table}[t]")
tables_tex.append(r"\centering")
tables_tex.append(r"\caption{\textbf{Calibration metrics.}}")
tables_tex.append(r"\label{tab:calibration}")
tables_tex.append(r"\begin{tabular}{lc}")
tables_tex.append(r"\toprule")
tables_tex.append(r"\textbf{Metric} & \textbf{Value} \\")
tables_tex.append(r"\midrule")
tables_tex.append(f"  ECE (Expected Calibration Error) & {ece:.4f} \\\\")
tables_tex.append(f"  Brier Score & {brier:.4f} \\\\")
tables_tex.append(f"  Accuracy & {final_metrics['classification']['accuracy']:.4f} \\\\")
tables_tex.append(f"  AUC & {final_metrics['classification']['auc']:.4f} \\\\")
tables_tex.append(f"  Sensitivity & {final_metrics['classification']['sensitivity']:.4f} \\\\")
tables_tex.append(f"  Specificity & {final_metrics['classification']['specificity']:.4f} \\\\")
tables_tex.append(r"\bottomrule")
tables_tex.append(r"\end{tabular}")
tables_tex.append(r"\end{table}")

with open(f'{OUT_DIR}/calibration_tables.tex', 'w') as f:
    f.write('\n'.join(tables_tex))
print(f"Saved: {OUT_DIR}/calibration_tables.tex")

print("\nDONE")
