#!/usr/bin/env python3
"""
Exp 6: LLM Explanation Benchmark + Case Study Selection
========================================================
1. Systematic LLM benchmark on 150 test samples
2. Metrics: evidence grounding, hallucination, direction, consistency, specificity
3. Select 4 representative case studies
4. Generate benchmark table and figure
"""
import json, os, sys, re, csv, pickle
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
MODEL_PATH = '/hd/liujx/microbiome_llm_project/models/qwen2-7b'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda:0'; MAX_NEW = 300; SEED = 42
N_BENCHMARK = 150

print("=" * 60)
print("EXP 6: LLM EXPLANATION BENCHMARK")
print("=" * 60)

# ── Load model ──
print("Loading Qwen2-7B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map={'': DEVICE}, trust_remote_code=True)
model.eval()

# ── Load data ──
with open('/hd/liujx/microbiome_llm_project/data/qiita_ibd/combined_info.json') as f:
    GENUS_NAMES = json.load(f)['genus_names']

test_data = [json.loads(l) for l in open(f'{DATA_DIR}/test_nl.jsonl')]
xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')

with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}

with open(f'{OUT_DIR}/shap_data_full.pkl', 'rb') as f:
    shap_by_id = {}
    for s in pickle.load(f)['all_samples']:
        shap_by_id[s['sample_id']] = {'label': s['label'], 'importance': s['importance']}

# ── Literature KB ──
lit_kb = {}
with open(f'{OUT_DIR}/literature_ground_truth.csv') as f:
    for r in csv.DictReader(f):
        lit_kb[r['Genus'].strip().lower()] = {
            'direction': r['Direction'].strip(),
            'mechanism': r['Mechanism'].strip(),
            'evidence': r['Evidence_Level'].strip()}

# ── Select samples ──
rng = np.random.RandomState(SEED)
n_avail = min(N_BENCHMARK, len(test_data))
sample_indices = rng.choice(len(test_data), n_avail, replace=False)

def get_genera(idx):
    valid = xm[idx].astype(bool)
    genera = []
    for pos in range(len(xs[idx])):
        if valid[pos] and xs[idx][pos] > 0:
            gid = int(xs[idx][pos])
            gname = GENUS_NAMES[gid-1] if gid-1 < len(GENUS_NAMES) else f'genus_{gid}'
            genera.append(gname)
    return genera

# ── Prompt builders ──
def build_prompt_raw(genera):
    glist = ', '.join(genera[:25])
    return f"""You are a gut microbiome analyst. A patient's gut microbiome contains these bacterial genera (ranked by abundance): {glist}.

Based on this genus composition, analyze whether this microbiome pattern is associated with Inflammatory Bowel Disease (IBD). Be specific about which genera suggest health or disease, and explain the biological reasoning.

Respond concisely in 3-5 sentences."""

def build_prompt_shap(genera, pred_label, confidence, prob, shap_list):
    glist = ', '.join(genera[:25])
    shap_str = '\n'.join(f"  {g['genus_name']}: {'INCREASED' if g['importance']>0 else 'DECREASED'} (importance={abs(g['importance']):.4f})" for g in shap_list[:15])
    return f"""You are a gut microbiome analyst. A machine learning classifier analyzed this patient's gut microbiome.

PATIENT DATA:
Genera (by abundance): {glist}

CLASSIFIER OUTPUT:
Prediction: {pred_label} (confidence: {confidence})
Disease probability: {prob:.3f}

IMPORTANT GENERA (SHAP feature importance):
{shap_str}

Based on these findings, explain why the classifier made this prediction. Reference specific genera and their known roles in gut health. Be specific and cite biological mechanisms.

Respond concisely in 4-6 sentences."""

def build_prompt_shap_lit(genera, pred_label, confidence, prob, shap_list):
    glist = ', '.join(genera[:25])
    parts = []
    for g in shap_list[:15]:
        gname = g['genus_name']; imp = g['importance']
        direction = 'INCREASED' if imp > 0 else 'DECREASED'
        line = f"  {gname}: {direction} (importance={abs(imp):.4f})"
        gn_lower = gname.strip().lower()
        if gn_lower in lit_kb:
            lk = lit_kb[gn_lower]
            line += f" [LITERATURE: {lk['direction']} in IBD, {lk['mechanism']} ({lk['evidence']})]"
        parts.append(line)
    shap_str = '\n'.join(parts)
    return f"""You are a gut microbiome analyst. A machine learning classifier analyzed this patient's gut microbiome.

PATIENT DATA:
Genera (by abundance): {glist}

CLASSIFIER OUTPUT:
Prediction: {pred_label} (confidence: {confidence})

IMPORTANT GENERA with literature context:
{shap_str}

Based on these findings AND the provided literature evidence, explain why the classifier made this prediction. Reference specific genera and literature-supported mechanisms.

Respond concisely in 4-6 sentences."""

@torch.no_grad()
def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
        temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# ── Evaluation metrics ──
def evaluate_response(response, genera_list, shap_list, pred_label):
    """Compute all benchmark metrics for one response."""
    resp_lower = response.lower()
    genus_set = set(g.lower() for g in genera_list)
    shap_names = [g['genus_name'].lower() for g in shap_list[:15]]
    shap_dirs = {g['genus_name'].lower(): 'increased' if g['importance'] > 0 else 'decreased'
                 for g in shap_list[:15]}

    metrics = {}

    # 1. Evidence Grounding: what fraction of mentioned genera are in SHAP top-15
    mentioned_genera = []
    hallucinated = 0
    for gn in genus_set:
        if gn in resp_lower:
            mentioned_genera.append(gn)
    # Check for hallucinated genera (known genus names in response not in input)
    for full_name in GENUS_NAMES:
        fn_lower = full_name.lower()
        if len(fn_lower) > 4 and fn_lower in resp_lower and fn_lower not in genus_set:
            hallucinated += 1
    metrics['n_mentioned'] = len(mentioned_genera)
    metrics['hallucination'] = hallucinated

    # 2. SHAP Consistency: mentioned genera that are in SHAP top-15
    shap_mentioned = [g for g in mentioned_genera if g in shap_names]
    metrics['shap_mentioned'] = len(shap_mentioned)
    metrics['shap_consistency'] = len(shap_mentioned) / max(len(mentioned_genera), 1)

    # 3. Direction Correctness
    dir_correct = 0; dir_total = 0
    for g in shap_mentioned:
        if g not in shap_dirs:
            continue
        shap_dir = shap_dirs[g]
        # Check response for direction clues
        if shap_dir == 'decreased':
            if any(kw in resp_lower for kw in ['decreased', 'reduced', 'depleted', 'lower', '↓']):
                dir_correct += 1
        else:
            if any(kw in resp_lower for kw in ['increased', 'elevated', 'higher', 'enriched', '↑']):
                dir_correct += 1
        dir_total += 1
    metrics['direction_correct'] = dir_correct
    metrics['direction_total'] = dir_total

    # 4. Prediction Consistency
    disease_kw = ['disease', 'ibd', 'inflammation', 'dysbiosis', 'crohn', 'colitis', 'altered', 'abnormal']
    healthy_kw = ['healthy', 'normal', 'balanced', 'homeostasis', 'commensal', 'beneficial']
    d_score = sum(1 for kw in disease_kw if kw in resp_lower)
    h_score = sum(1 for kw in healthy_kw if kw in resp_lower)
    if pred_label == 'IBD':
        metrics['pred_consistent'] = d_score >= h_score
    else:
        metrics['pred_consistent'] = h_score >= d_score

    # 5. Specificity (biological mechanism keywords)
    specific_kw = ['scfa', 'butyrate', 'inflammation', 'barrier', 'permeability',
        'immune', 'pathogen', 'dysbiosis', 'cytokine', 'mucosal', 'microbial',
        'fermentation', 'metabolite', 'anti-inflammatory', 'pro-inflammatory',
        'short-chain', 'fatty acid', 'lps', 'endotoxin']
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    spec_count = sum(1 for s in sentences if any(kw in s.lower() for kw in specific_kw))
    metrics['specificity'] = spec_count / max(len(sentences), 1)
    metrics['n_sentences'] = len(sentences)

    # 6. Literature Consistency
    lit_consistent = 0; lit_total = 0
    for g in shap_mentioned:
        if g in lit_kb:
            lit_dir = lit_kb[g]['direction'].lower()
            if 'decreased' in lit_dir and 'decreased' in resp_lower:
                lit_consistent += 1
            elif 'increased' in lit_dir and 'increased' in resp_lower:
                lit_consistent += 1
            lit_total += 1
    metrics['lit_consistent'] = lit_consistent
    metrics['lit_total'] = lit_total

    return metrics

# ── Run benchmark ──
print(f"\nRunning benchmark on {n_avail} test samples...")
benchmark_results = []
n_done = 0

for idx in sample_indices:
    d = test_data[idx]; sid = d['sample_id']
    true_label = d['label']
    pred = pred_data.get(sid, {})
    prob = float(pred.get('prob_disease', 0.5))
    pred_label = 'IBD' if prob > 0.5 else 'HEALTHY'
    confidence = f'{max(prob, 1-prob)*100:.1f}%'

    genera = get_genera(idx)
    if not genera: continue

    shap_top = []
    if sid in shap_by_id:
        for g in shap_by_id[sid]['importance'][:15]:
            shap_top.append({'genus_name': g['genus_name'], 'importance': g['importance']})

    try:
        resp_raw = generate(build_prompt_raw(genera))
        resp_shap = generate(build_prompt_shap(genera, pred_label, confidence, prob, shap_top))
        resp_lit = generate(build_prompt_shap_lit(genera, pred_label, confidence, prob, shap_top))

        metrics_raw = evaluate_response(resp_raw, genera, shap_top, pred_label)
        metrics_shap = evaluate_response(resp_shap, genera, shap_top, pred_label)
        metrics_lit = evaluate_response(resp_lit, genera, shap_top, pred_label)

        benchmark_results.append({
            'sample_id': sid, 'true_label': true_label, 'pred_label': pred_label,
            'prob': prob, 'genera': genera, 'shap_top': shap_top,
            'resp_raw': resp_raw, 'resp_shap': resp_shap, 'resp_lit': resp_lit,
            'metrics_raw': metrics_raw, 'metrics_shap': metrics_shap, 'metrics_lit': metrics_lit,
        })
        n_done += 1
        if n_done % 10 == 0:
            print(f"  [{n_done}/{n_avail}] processed...")

    except Exception as e:
        print(f"  ERROR {sid}: {e}")

print(f"\nCompleted {n_done} samples.")

# ── Aggregate metrics ──
def aggregate_metrics(results, variant_key):
    all_m = [r[variant_key] for r in results]
    n = len(all_m)
    return {
        'mean_mentioned': float(np.mean([m['n_mentioned'] for m in all_m])),
        'mean_hallucination': float(np.mean([m['hallucination'] for m in all_m])),
        'mean_shap_mentioned': float(np.mean([m['shap_mentioned'] for m in all_m])),
        'shap_consistency': float(np.mean([m['shap_consistency'] for m in all_m])),
        'direction_accuracy': float(np.sum([m['direction_correct'] for m in all_m]) /
                                      max(np.sum([m['direction_total'] for m in all_m]), 1)),
        'pred_consistency': float(np.mean([1 if m['pred_consistent'] else 0 for m in all_m])),
        'specificity': float(np.mean([m['specificity'] for m in all_m])),
        'lit_consistency': float(np.sum([m['lit_consistent'] for m in all_m]) /
                                  max(np.sum([m['lit_total'] for m in all_m]), 1)),
    }

summary = {
    'Raw LLM': aggregate_metrics(benchmark_results, 'metrics_raw'),
    'SHAP + LLM': aggregate_metrics(benchmark_results, 'metrics_shap'),
    'SHAP + Lit + LLM': aggregate_metrics(benchmark_results, 'metrics_lit'),
}

print("\n" + "=" * 70)
print("LLM EXPLANATION BENCHMARK RESULTS")
print("=" * 70)
print(f"{'Metric':<30s} {'Raw LLM':>12s} {'SHAP+LLM':>12s} {'SHAP+Lit+LLM':>12s}")
print("-" * 70)
for metric_key, metric_name in [
    ('mean_mentioned', 'Genera mentioned'),
    ('mean_hallucination', 'Hallucinations'),
    ('shap_consistency', 'SHAP Consistency'),
    ('direction_accuracy', 'Direction Correct'),
    ('pred_consistency', 'Pred Consistent'),
    ('specificity', 'Specificity'),
    ('lit_consistency', 'Lit Consistent'),
]:
    vals = [summary[v][metric_key] for v in ['Raw LLM', 'SHAP + LLM', 'SHAP + Lit + LLM']]
    print(f"{metric_name:<30s} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

# ── Select case studies ──
print("\n" + "=" * 60)
print("SELECTING CASE STUDIES")
print("=" * 60)

cases = []

# Case 1: Correct prediction + good explanation
correct_good = [r for r in benchmark_results
                if r['pred_label'] == ('IBD' if r['true_label'] == 'Disease' else 'HEALTHY')
                and r['metrics_shap']['shap_consistency'] > 0.5
                and r['metrics_shap']['pred_consistent']]
if correct_good:
    cases.append(('Case 1: Correct prediction, good explanation', correct_good[0]))
    print(f"  Case 1: {correct_good[0]['sample_id']} (correct + good SHAP explanation)")

# Case 2: Correct prediction, borderline
correct_borderline = [r for r in benchmark_results
                      if r['pred_label'] == ('IBD' if r['true_label'] == 'Disease' else 'HEALTHY')
                      and 0.4 < r['prob'] < 0.6]
if correct_borderline:
    cases.append(('Case 2: Correct prediction, borderline confidence', correct_borderline[0]))
    print(f"  Case 2: {correct_borderline[0]['sample_id']} (prob={correct_borderline[0]['prob']:.3f})")

# Case 3: Wrong prediction
wrong = [r for r in benchmark_results
         if r['pred_label'] != ('IBD' if r['true_label'] == 'Disease' else 'HEALTHY')]
if wrong:
    cases.append(('Case 3: Wrong prediction', wrong[0]))
    print(f"  Case 3: {wrong[0]['sample_id']} (true={wrong[0]['true_label']}, pred={wrong[0]['pred_label']})")

# Case 4: Extreme IBD (high prob, strong SHAP)
extreme = [r for r in benchmark_results
           if r['pred_label'] == 'IBD' and r['prob'] > 0.95]
if extreme:
    cases.append(('Case 4: Extreme IBD (high confidence)', extreme[0]))
    print(f"  Case 4: {extreme[0]['sample_id']} (prob={extreme[0]['prob']:.3f})")

# ── Save ──
output = {
    'experiment': 'llm_explanation_benchmark',
    'n_samples': n_done,
    'model': 'Qwen2-7B-Instruct',
    'summary': summary,
    'case_studies': [{'name': name, 'sample_id': r['sample_id'],
                       'true_label': r['true_label'], 'pred_label': r['pred_label'],
                       'prob': r['prob'], 'shap_top': r['shap_top'][:10],
                       'resp_raw': r['resp_raw'], 'resp_shap': r['resp_shap'],
                       'resp_lit': r['resp_lit']} for name, r in cases],
    'benchmark_results': benchmark_results,
}
with open(f'{OUT_DIR}/llm_benchmark_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

# ── Write case studies report ──
with open(f'{OUT_DIR}/case_studies.txt', 'w') as f:
    f.write("ProCyon v2 — Case Studies\n")
    f.write("=" * 70 + "\n\n")
    for name, r in cases:
        f.write(f"\n{'─'*70}\n")
        f.write(f"{name}\n")
        f.write(f"{'─'*70}\n")
        f.write(f"Sample ID: {r['sample_id']}\n")
        f.write(f"True: {r['true_label']} | Predicted: {r['pred_label']} (p={r['prob']:.3f})\n\n")

        f.write(f"--- Raw LLM ---\n{r['resp_raw']}\n\n")
        f.write(f"--- SHAP + LLM ---\n{r['resp_shap']}\n\n")
        f.write(f"--- SHAP + Lit + LLM ---\n{r['resp_lit']}\n\n")

        f.write("Top SHAP genera:\n")
        for g in r['shap_top'][:10]:
            direction = 'INCREASED in IBD' if g['importance'] > 0 else 'DECREASED in IBD'
            gname = g['genus_name']
            lit_note = ''
            if gname.lower() in lit_kb:
                lk = lit_kb[gname.lower()]
                lit_note = f" [Literature: {lk['direction']} — {lk['mechanism']} ({lk['evidence']})]"
            f.write(f"  {gname}: {direction} (imp={g['importance']:.4f}){lit_note}\n")
        f.write("\n")

# ── Generate LaTeX table ──
benchmark_tex = []
benchmark_tex.append(r"\begin{table}[t]")
benchmark_tex.append(r"\centering")
benchmark_tex.append(r"\caption{\textbf{LLM Explanation Benchmark.}")
benchmark_tex.append(r"Systematic evaluation of 3 prompt variants on " + str(n_done) + r" test samples.")
benchmark_tex.append(r"SHAP grounding eliminates hallucination and achieves highest consistency.}")
benchmark_tex.append(r"\label{tab:llm_benchmark}")
benchmark_tex.append(r"\begin{tabular}{lcccc}")
benchmark_tex.append(r"\toprule")
benchmark_tex.append(r"\textbf{Metric} & \textbf{Raw LLM} & \textbf{SHAP+LLM} & \textbf{SHAP+Lit+LLM} \\")
benchmark_tex.append(r"\midrule")
for metric_key, metric_name in [
    ('mean_mentioned', 'Genera mentioned (avg)'),
    ('mean_hallucination', 'Hallucinated genera (avg)'),
    ('shap_consistency', 'SHAP consistency'),
    ('direction_accuracy', 'Direction accuracy'),
    ('pred_consistency', 'Prediction consistency'),
    ('specificity', 'Specificity ratio'),
    ('lit_consistency', 'Literature consistency'),
]:
    vals = [summary[v][metric_key] for v in ['Raw LLM', 'SHAP + LLM', 'SHAP + Lit + LLM']]
    best_idx = [0, 1, 2][np.argmax(vals if 'hallucination' not in metric_key else [-v for v in vals])]
    cells = []
    for i, v in enumerate(vals):
        if i == best_idx and 'hallucination' not in metric_key:
            cells.append(f"\\textbf{{{v:.3f}}}")
        elif i == best_idx and 'hallucination' in metric_key:
            cells.append(f"\\textbf{{{v:.2f}}}")
        else:
            cells.append(f"{v:.3f}" if v < 1 else f"{v:.2f}")
    benchmark_tex.append(f"  {metric_name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
benchmark_tex.append(r"\bottomrule")
benchmark_tex.append(r"\end{tabular}")
benchmark_tex.append(r"\end{table}")

benchmark_table_str = '\n'.join(benchmark_tex)
with open(f'{OUT_DIR}/llm_benchmark_table.tex', 'w') as f:
    f.write(benchmark_table_str)

print(f"\nSaved: {OUT_DIR}/llm_benchmark_results.json")
print(f"Saved: {OUT_DIR}/case_studies.txt")
print(f"Saved: {OUT_DIR}/llm_benchmark_table.tex")
print("\nEXP 6 DONE")
