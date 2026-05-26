#!/usr/bin/env python3
"""Run evaluation on trained Qwen2-0.5B model (after pipeline training completed)"""
import os, re, json, sys
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAINED_PATH = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_qa"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b"
os.makedirs(EVAL_DIR, exist_ok=True)

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_label(text):
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        return m.group(1).strip('。，, \n')
    for kw in ['Healthy', 'IBD', 'CD', 'UC']:
        if kw in text:
            return kw
    cn_map = {'健康': 'Healthy', '炎症性肠病': 'IBD', '克罗恩': 'CD', '溃疡性结肠炎': 'UC'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None

def evaluate(model, tokenizer, test_data, device, name="model", max_new_tokens=64):
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        messages = item['messages']
        true_label = item['label']
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted_label = extract_label(generated)
        predictions.append({
            'sample_id': item.get('sample_id', ''),
            'true_label': true_label,
            'predicted_label': predicted_label or 'UNKNOWN',
            'generated': generated.strip()[:200],
        })
        true_labels.append(true_label)
        pred_labels.append(predicted_label or 'UNKNOWN')
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(test_data)}")

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    all_labels = ['Healthy', 'IBD', 'CD', 'UC']
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=all_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='weighted', zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    header = f"{'':>12}"
    for l in all_labels:
        header += f" {l:>10}"
    print(header)
    for i, label in enumerate(all_labels):
        row = f"{label:>10}:"
        for j in range(4):
            row += f" {cm[i][j]:>10}"
        print(row)

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'predictions': predictions}

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

test_data = load_jsonl(TEST_DATA)
print(f"Test samples: {len(test_data)}")
print(f"Label distribution: {dict(Counter(d['label'] for d in test_data))}")

# Load trained model
print("\nLoading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": "cuda:0"}, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, TRAINED_PATH)
model.config.use_cache = True
print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

results = evaluate(model, tokenizer, test_data, device, "LoRA Fine-tuned Qwen2-0.5B")

# Save
with open(os.path.join(EVAL_DIR, 'predictions_after.json'), 'w') as f:
    json.dump(results['predictions'], f, indent=2, ensure_ascii=False)
with open(os.path.join(EVAL_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EVAL_DIR}/")
print("✅ Done")
