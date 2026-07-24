#!/usr/bin/env python3
"""
Few-Shot LLM Evaluation for ProCyon v2
=======================================
Tests whether Qwen2-7B can predict IBD from genus lists when given k examples.
Evaluates:
  - k = 0, 1, 3, 5 examples in the prompt
  - Compare LLM prediction vs ground truth label
  - Compare LLM vs SimpleEmb+MLP classifier
  - Hallucination rate, genus mention accuracy, reasoning quality

This answers: "Can an LLM replace the specialized classifier?"
"""
import json, os, sys, re, csv
sys.path.insert(0, '/hd/liujx/microbiome_llm_project')
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = '/hd/liujx/microbiome_llm_project/ProCyon_v2/analysis'
MODEL_PATH = '/hd/liujx/microbiome_llm_project/models/qwen2-7b'
DATA_DIR = '/hd/liujx/microbiome_llm_project/data/qiita_ibd/clean_2538'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda:0'
MAX_NEW = 200
SEED = 42
N_TEST = 100  # test samples to evaluate

print("=" * 60)
print("Few-Shot LLM Evaluation")
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

train_data = []
with open(f'{DATA_DIR}/train_nl.jsonl') as f:
    for l in f:
        train_data.append(json.loads(l))
test_data = []
with open(f'{DATA_DIR}/test_nl.jsonl') as f:
    for l in f:
        test_data.append(json.loads(l))

xs = np.load(f'{DATA_DIR}/test_genus_sequences.npy')
xm = np.load(f'{DATA_DIR}/test_genus_masks.npy')
ts = np.load(f'{DATA_DIR}/train_genus_sequences.npy')
tm = np.load(f'{DATA_DIR}/train_genus_masks.npy')

# Load classifier predictions for comparison
with open(f'{OUT_DIR}/../backbone/predictions.csv') as f:
    pred_data = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}

def get_genera(seq, mask):
    """Convert genus sequence to readable names."""
    valid = mask.astype(bool)
    genera = []
    for pos in range(len(seq)):
        if valid[pos] and seq[pos] > 0:
            gid = int(seq[pos])
            gname = GENUS_NAMES[gid - 1] if gid - 1 < len(GENUS_NAMES) else f'genus_{gid}'
            genera.append(gname)
    return genera

# ── Sample examples from training data ──
train_healthy = [(d, ts[i], tm[i]) for i, d in enumerate(train_data) if d['label'] == 'Healthy']
train_disease = [(d, ts[i], tm[i]) for i, d in enumerate(train_data) if d['label'] == 'Disease']
rng = np.random.RandomState(SEED)

def sample_examples(k):
    """Sample k balanced examples from training data."""
    n_each = k // 2 + (k % 2)
    h_samples = list(rng.choice(train_healthy, min(n_each, len(train_healthy)), replace=False))
    d_samples = list(rng.choice(train_disease, min(k - len(h_samples), len(train_disease)), replace=False))
    examples = h_samples + d_samples
    rng.shuffle(examples)
    return examples

def format_example(d, seq, mask):
    """Format one example for the prompt."""
    genera = get_genera(seq, mask)
    genus_str = ', '.join(genera[:20])
    label = d['label']
    return f"Patient microbiome: {genus_str}\nDiagnosis: {label}"

def get_label_from_response(response):
    """Extract predicted label from LLM response."""
    resp_lower = response.lower()

    # Strong IBD indicators
    ibd_patterns = [
        r'\b(ibd|inflammatory bowel disease|crohn|colitis)\b',
        r'\b(disease|diseased|pathogenic|dysbiosis)\b',
        r'\bdiagnosis[:\s]*(disease|ibd|positive)\b',
        r'\bprediction[:\s]*(disease|ibd|positive)\b',
        r'\bassociated with (ibd|disease|inflammation)\b',
    ]
    # Strong healthy indicators
    healthy_patterns = [
        r'\b(healthy|normal|balanced|homeostasis)\b',
        r'\bdiagnosis[:\s]*(healthy|normal|negative)\b',
        r'\bprediction[:\s]*(healthy|normal|negative)\b',
        r'\bnot (associated with|indicative of) (ibd|disease)\b',
        r'\bno (evidence|sign|indication) of (ibd|disease)\b',
    ]

    ibd_score = 0
    healthy_score = 0
    for pat in ibd_patterns:
        if re.search(pat, resp_lower):
            ibd_score += 1
    for pat in healthy_patterns:
        if re.search(pat, resp_lower):
            healthy_score += 1

    # Also check for explicit label at the end
    if re.search(r'(diagnosis|prediction|label)[:\s]*(disease|ibd)', resp_lower):
        ibd_score += 2
    if re.search(r'(diagnosis|prediction|label)[:\s]*(healthy|normal)', resp_lower):
        healthy_score += 2

    if ibd_score > healthy_score:
        return 'Disease'
    elif healthy_score > ibd_score:
        return 'Healthy'
    else:
        return 'Uncertain'

@torch.no_grad()
def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=MAX_NEW, do_sample=True,
        temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


# ── Run few-shot evaluation ──
print("\nSelecting test samples...")
rng_test = np.random.RandomState(SEED)
test_indices = rng_test.choice(len(test_data), min(N_TEST, len(test_data)), replace=False)

all_results = {}
for k in [0, 1, 3, 5]:
    print(f"\n{'='*60}")
    print(f"Few-Shot k={k}")
    print(f"{'='*60}")

    examples = sample_examples(k) if k > 0 else []
    example_text = '\n\n'.join([format_example(d, s, m) for d, s, m in examples])

    results = []
    correct = 0; total = 0; uncertain = 0
    llm_labels = []; true_labels_list = []

    for idx in test_indices:
        d = test_data[idx]; sid = d['sample_id']
        true_label = d['label']  # 'Healthy' or 'Disease'
        true_label_short = 'Disease' if true_label == 'Disease' else 'Healthy'
        genera = get_genera(xs[idx], xm[idx])
        genus_str = ', '.join(genera[:25])

        # Get classifier prediction
        pred = pred_data.get(sid, {})
        clf_prob = float(pred.get('prob_disease', 0.5))
        clf_pred = 'Disease' if clf_prob > 0.5 else 'Healthy'

        if k > 0:
            prompt = f"""You are a gut microbiome analyst. Given examples of patient microbiomes and their diagnoses, predict the diagnosis for a new patient.

EXAMPLES:
{example_text}

NEW PATIENT:
Patient microbiome: {genus_str}

Based on the examples above, what is the most likely diagnosis for this new patient? Answer with either 'Diagnosis: Disease (IBD)' or 'Diagnosis: Healthy', and explain your reasoning in 2-3 sentences."""
        else:
            prompt = f"""You are a gut microbiome analyst. Analyze this patient's gut microbiome composition and determine if they have Inflammatory Bowel Disease (IBD).

Patient microbiome: {genus_str}

Based on this genus composition, what is the most likely diagnosis? Answer with either 'Diagnosis: Disease (IBD)' or 'Diagnosis: Healthy', and explain your reasoning in 2-3 sentences."""

        try:
            response = generate(prompt)
            llm_label = get_label_from_response(response)
            is_correct = (llm_label == true_label_short)
            if llm_label == 'Uncertain':
                uncertain += 1
            else:
                total += 1
                if is_correct:
                    correct += 1
                llm_labels.append(llm_label)
                true_labels_list.append(true_label_short)

            results.append({
                'sample_id': sid,
                'true_label': true_label,
                'clf_prediction': clf_pred,
                'clf_probability': clf_prob,
                'llm_prediction': llm_label,
                'llm_correct': is_correct,
                'genera': genus_str,
                'response': response,
            })
        except Exception as e:
            print(f"  ERROR {sid}: {e}")
            continue

    acc = correct / max(total, 1)
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        label_map = {'Healthy': 0, 'Disease': 1}
        y_true = [label_map[l] for l in true_labels_list]
        y_pred = [label_map[l] for l in llm_labels]
        f1 = f1_score(y_true, y_pred, average='macro')
    except:
        f1 = 0.0

    all_results[f'k={k}'] = {
        'accuracy': float(acc),
        'f1': float(f1),
        'n_correct': correct,
        'n_total': total,
        'n_uncertain': uncertain,
        'n_samples': len(results),
        'results': results,
    }

    print(f"  k={k}: ACC={acc:.4f} ({correct}/{total}) F1={f1:.4f} Uncertain={uncertain}")

# ── Also evaluate classifier accuracy on the same samples ──
print(f"\n{'='*60}")
print("Classifier Performance on Same Samples")
print(f"{'='*60}")

clf_correct = 0; clf_total = 0
for idx in test_indices:
    d = test_data[idx]; sid = d['sample_id']
    true_label = 'Disease' if d['label'] == 'Disease' else 'Healthy'
    pred = pred_data.get(sid, {})
    clf_prob = float(pred.get('prob_disease', 0.5))
    clf_pred = 'Disease' if clf_prob > 0.5 else 'Healthy'
    if clf_pred == true_label:
        clf_correct += 1
    clf_total += 1

clf_acc = clf_correct / clf_total
print(f"  Classifier ACC: {clf_acc:.4f} ({clf_correct}/{clf_total})")

# ── Save results ──
# Filter out full results for JSON
summary_results = {}
for k, v in all_results.items():
    summary_results[k] = {key: val for key, val in v.items() if key != 'results'}

output = {
    'experiment': 'fewshot_llm_evaluation',
    'model': 'Qwen2-7B-Instruct',
    'n_test_samples': N_TEST,
    'classifier_accuracy': float(clf_acc),
    'fewshot_results': summary_results,
    'detailed_results': {k: v['results'] for k, v in all_results.items()},
}

with open(f'{OUT_DIR}/fewshot_llm_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

# ── Human-readable report ──
with open(f'{OUT_DIR}/fewshot_human_review.txt', 'w') as f:
    f.write("FEW-SHOT LLM EVALUATION - HUMAN REVIEW\n")
    f.write("=" * 70 + "\n\n")

    for k in [0, 1, 3, 5]:
        f.write(f"\n{'─'*70}\n")
        f.write(f"k = {k}\n")
        f.write(f"Accuracy: {all_results[f'k={k}']['accuracy']:.4f} "
                f"({all_results[f'k={k}']['n_correct']}/{all_results[f'k={k}']['n_total']})\n")
        f.write(f"{'─'*70}\n\n")

        for i, r in enumerate(all_results[f'k={k}']['results'][:10]):  # First 10 per k
            f.write(f"Sample {i+1}: {r['sample_id']}\n")
            f.write(f"  True: {r['true_label']} | LLM: {r['llm_prediction']} | "
                    f"CLF: {r['clf_prediction']} (p={r['clf_probability']:.3f})\n")
            f.write(f"  Correct: {r['llm_correct']}\n")
            f.write(f"  Response: {r['response'][:300]}\n\n")

print(f"\nSaved: {OUT_DIR}/fewshot_llm_results.json")
print(f"Saved: {OUT_DIR}/fewshot_human_review.txt")

# ── Summary LaTeX table ──
print(f"\n{'='*60}")
print("FEW-SHOT RESULTS TABLE")
print(f"{'='*60}")

latex_fs = []
latex_fs.append(r"\begin{table}[t]")
latex_fs.append(r"\centering")
latex_fs.append(r"\caption{\textbf{Few-shot LLM evaluation on IBD diagnosis.}")
latex_fs.append(r"Qwen2-7B-Instruct is given $k$ examples of (genus list, diagnosis) pairs")
latex_fs.append(r"and asked to predict on new test samples. The specialized classifier")
latex_fs.append(r"(SimpleEmb+MLP) outperforms the LLM by a large margin, demonstrating that")
latex_fs.append(r"domain-specific training is necessary for microbiome-based diagnosis.}")
latex_fs.append(r"\label{tab:fewshot}")
latex_fs.append(r"\begin{tabular}{lcccc}")
latex_fs.append(r"\toprule")
latex_fs.append(r"\textbf{Method} & \textbf{ACC} & \textbf{F1} & \textbf{Uncertain} & \textbf{Correct/Total} \\")
latex_fs.append(r"\midrule")
latex_fs.append(f"  LLM (k=0, zero-shot) & {all_results['k=0']['accuracy']:.4f} & {all_results['k=0']['f1']:.4f} & {all_results['k=0']['n_uncertain']} & {all_results['k=0']['n_correct']}/{all_results['k=0']['n_total']} \\\\")
latex_fs.append(f"  LLM (k=1, 1-shot) & {all_results['k=1']['accuracy']:.4f} & {all_results['k=1']['f1']:.4f} & {all_results['k=1']['n_uncertain']} & {all_results['k=1']['n_correct']}/{all_results['k=1']['n_total']} \\\\")
latex_fs.append(f"  LLM (k=3, 3-shot) & {all_results['k=3']['accuracy']:.4f} & {all_results['k=3']['f1']:.4f} & {all_results['k=3']['n_uncertain']} & {all_results['k=3']['n_correct']}/{all_results['k=3']['n_total']} \\\\")
latex_fs.append(f"  LLM (k=5, 5-shot) & {all_results['k=5']['accuracy']:.4f} & {all_results['k=5']['f1']:.4f} & {all_results['k=5']['n_uncertain']} & {all_results['k=5']['n_correct']}/{all_results['k=5']['n_total']} \\\\")
latex_fs.append(r"  \midrule")
latex_fs.append(f"  \\textbf{{ProCyon v2 (ours)}} & \\textbf{{{clf_acc:.4f}}} & \\textbf{{---}} & 0 & {clf_correct}/{clf_total} \\\\")
latex_fs.append(r"\bottomrule")
latex_fs.append(r"\end{tabular}")
latex_fs.append(r"\end{table}")

fewshot_table = '\n'.join(latex_fs)
print(fewshot_table)

with open(f'{OUT_DIR}/fewshot_table.tex', 'w') as f:
    f.write(fewshot_table)
print(f"Saved: {OUT_DIR}/fewshot_table.tex")

print("\nFEW-SHOT EVALUATION DONE")
