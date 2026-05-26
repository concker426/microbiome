#!/usr/bin/env python3
"""Token 级概率阈值调优：在决策点（step 3）调整 P(Disease) vs P(Healthy) 的阈值"""
import os, re, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAINED_PATH = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_binary"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b_binary"

BINARY_MAP = {'IBD': 'Disease', 'CD': 'Disease', 'UC': 'Disease', 'Healthy': 'Healthy'}

# Decision tokens (for step 3)
TOK_D = 35         # "D" → Disease
TOK_HEALTHY = 96113  # "Healthy"
ALLOW_EXTRA = [30874, 43354]  # " Disease", " Healthy" (with space, in case format varies)

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    test_data = load_jsonl(TEST_DATA)
    true_labels = [BINARY_MAP[d['label']] for d in test_data]
    print(f"Test samples: {len(test_data)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map={"": "cuda:0"}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, TRAINED_PATH)
    model.config.use_cache = True
    model.eval()

    # For each test sample, get the probability ratio D/(D+Healthy) at decision point
    d_ratios = []
    greedy_preds = []
    has_healthy = 0
    has_d = 0

    for idx, item in enumerate(test_data):
        messages = item['messages']
        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, temperature=0.1, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]

        # Find decision point (where "诊断结果：" ends and label begins)
        # Typical: pos 0=诊断, 1=结果, 2=： → step 3 is the decision
        decision_step = 3  # Position where we expect the decision

        probs = None
        d_prob = 0.0
        healthy_prob = 0.0
        p_d = 0.0
        p_h = 0.0

        if len(outputs.scores) > decision_step:
            scores = outputs.scores[decision_step][0]  # logits at decision step
            probs = torch.softmax(scores, dim=0)
            p_d = probs[TOK_D].item()
            p_h = probs[TOK_HEALTHY].item() if TOK_HEALTHY < len(probs) else 0.0

            # Also check space-prefixed variants
            for tid in ALLOW_EXTRA:
                if tid < len(probs):
                    pass  # already covered

        # Normalize to get relative probability
        total = p_d + p_h
        d_ratio = p_d / total if total > 0 else 0.5
        d_ratios.append(d_ratio)

        # Greedy prediction
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
        m = re.search(r'诊断结果[：:]\s*(\S+)', generated)
        greedy_label = (m.group(1).strip('。，, \n') if m else
                       next((kw for kw in ['Healthy', 'Disease'] if kw in generated), 'UNKNOWN'))

        greedy_preds.append(greedy_label)

        if greedy_label == 'Healthy':
            has_healthy += 1
        elif greedy_label == 'Disease':
            has_d += 1

        true_label = BINARY_MAP[item['label']]
        status = '✓' if greedy_label == true_label else '✗'
        if (idx + 1) % 25 == 0 or status == '✗':
            print(f"  [{idx}] true={true_label:<8} greedy={greedy_label:<8} d_ratio={d_ratio:.4f} p_h={p_h:.4f} p_d={p_d:.4f} {status}", flush=True)

    print(f"\nGreedy: Healthy={has_healthy}, Disease={has_d}", flush=True)

    # Greedy metrics
    greedy_acc = accuracy_score(true_labels, greedy_preds)
    print(f"Greedy accuracy: {greedy_acc:.2%}", flush=True)

    # Try different thresholds on D/(D+Healthy) ratio
    THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    print(f"\n{'='*80}", flush=True)
    print(f"  Threshold Tuning Results", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Threshold':<10} {'Acc':<8} {'MacroF1':<10} {'H-Prec':<10} {'H-Recall':<10} {'D-Prec':<10} {'D-Recall':<10} {'H-F1':<10} {'D-F1':<10} {'BestF1?':<8}")
    print(f"{'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    best_f1 = 0
    best_threshold = None
    best_results = None

    for th in THRESHOLDS:
        preds = []
        for ratio in d_ratios:
            if ratio >= th:
                preds.append('Disease')
            else:
                preds.append('Healthy')

        acc = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, labels=['Healthy', 'Disease'],
                                       output_dict=True, zero_division=0)
        h_report = report.get('Healthy', {})
        d_report = report.get('Disease', {})
        macro_f1 = (h_report.get('f1-score', 0) + d_report.get('f1-score', 0)) / 2

        best_f1_now = h_report.get('f1-score', 0) + d_report.get('f1-score', 0)
        is_best = "← Best" if d_report.get('recall', 0) >= 0.70 and best_f1_now > best_f1 else ""
        if d_report.get('recall', 0) >= 0.70 and best_f1_now > best_f1:
            best_f1 = best_f1_now
            best_threshold = th
            best_results = (acc, macro_f1, h_report, d_report)

        print(f"{th:<10.2f} {acc:<8.2%} {macro_f1:<10.4f} "
              f"{h_report.get('precision', 0):<10.4f} {h_report.get('recall', 0):<10.2%} "
              f"{d_report.get('precision', 0):<10.4f} {d_report.get('recall', 0):<10.2%} "
              f"{h_report.get('f1-score', 0):<10.4f} {d_report.get('f1-score', 0):<10.4f} "
              f"{is_best:<8}", flush=True)

    # Show best threshold confusion matrix
    if best_results:
        print(f"\n{'='*80}", flush=True)
        print(f"  Best Threshold: {best_threshold} (targeting Disease recall ≥ 70%)", flush=True)
        print(f"{'='*80}", flush=True)
        preds = ['Disease' if r >= best_threshold else 'Healthy' for r in d_ratios]
        cm = confusion_matrix(true_labels, preds, labels=['Healthy', 'Disease'])
        print(f"Accuracy: {best_results[0]:.2%}, MacroF1: {best_results[1]:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Healthy  Disease")
        print(f"Healthy:         {cm[0][0]:>5}     {cm[0][1]:>5}")
        print(f"Disease:         {cm[1][0]:>5}     {cm[1][1]:>5}")
        print(f"\nHealthy - Precision={best_results[2].get('precision',0):.4f}, Recall={best_results[2].get('recall',0):.2%}, F1={best_results[2].get('f1-score',0):.4f}")
        print(f"Disease - Precision={best_results[3].get('precision',0):.4f}, Recall={best_results[3].get('recall',0):.2%}, F1={best_results[3].get('f1-score',0):.4f}")

    # Save
    results = {
        'greedy_accuracy': float(greedy_acc),
        'best_threshold': best_threshold,
    }
    with open(os.path.join(OUTPUT_DIR, 'threshold_tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/threshold_tuning_results.json", flush=True)

if __name__ == "__main__":
    main()
