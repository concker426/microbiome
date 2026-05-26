#!/usr/bin/env python3
"""Prompt 调优：保留原始对话格式，仅微调指令部分"""
import os, re, json
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAINED_PATH = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_binary"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b_binary"

BINARY_MAP = {'IBD': 'Disease', 'CD': 'Disease', 'UC': 'Disease', 'Healthy': 'Healthy'}

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_label(text, valid_labels=None):
    if valid_labels is None:
        valid_labels = ['Healthy', 'Disease']
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in valid_labels:
            return label
    for kw in valid_labels:
        if kw in text:
            return kw
    return None

# Instruction variations (only modify the instruction sentence, keep OTU data unchanged)
INSTRUCTION_VARIANTS = {
    "baseline": "请判断该样本的健康状态（Healthy或Disease），并简要说明理由。",
    "disease_first": "请判断该样本的健康状态（Healthy或Disease）。注意：该数据集中Disease（IBD、CD、UC）样本占多数，如怀疑请优先考虑Disease。",
    "low_threshold": "请判断该样本的健康状态（Healthy或Disease）。如果你发现任何菌群异常迹象，请判断为Disease。宁可误报，不可漏诊。",
    "balanced": "请判断该样本的健康状态（Healthy或Disease）。Healthy和Disease的判断标准请保持平衡。",
    "strict_healthy": "请判断该样本的健康状态（Healthy或Disease）。仅当菌群构成完全正常时才判断为Healthy，有任何异常请判断为Disease。",
}

def make_variant_messages(orig_messages, instruction):
    """替换原 user message 中的指令句子，其余完全不变"""
    user_content = orig_messages[0]['content']
    # The last sentence is the instruction - replace it
    lines = user_content.rsplit('\n\n', 1)
    if len(lines) == 2:
        new_user_content = lines[0] + '\n\n' + instruction
    else:
        new_user_content = user_content
    return [{"role": "user", "content": new_user_content}]

def evaluate_variant(model, tokenizer, test_data, device, instruction, gen_config, name=""):
    model.eval()
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        orig_messages = item['messages']
        true_label = BINARY_MAP[item['label']]
        messages = make_variant_messages(orig_messages, instruction)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 64),
                temperature=gen_config.get("temperature", 0.1),
                do_sample=gen_config.get("do_sample", False),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted_label = extract_label(generated)
        true_labels.append(true_label)
        pred_labels.append(predicted_label or 'UNKNOWN')

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(test_data)}", flush=True)

    all_labels = ['Healthy', 'Disease']
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0)
    report = classification_report(true_labels, pred_labels, labels=all_labels, output_dict=True, zero_division=0)

    return {
        'name': name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'healthy_recall': report.get('Healthy', {}).get('recall', 0),
        'disease_recall': report.get('Disease', {}).get('recall', 0),
        'healthy_f1': report.get('Healthy', {}).get('f1-score', 0),
        'disease_f1': report.get('Disease', {}).get('f1-score', 0),
        'confusion_matrix': cm.tolist(),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    test_data = load_jsonl(TEST_DATA)
    print(f"Test samples: {len(test_data)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map={"": "cuda:0"}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, TRAINED_PATH)
    model.config.use_cache = True

    # Only test greedy (do_sample=False) for all instruction variants
    # Since temperature/do_sample flags are ignored anyway
    gen_config = {"temperature": 0.1, "do_sample": False, "max_new_tokens": 64}

    results = []
    for iname, instruction in INSTRUCTION_VARIANTS.items():
        name = f"instruct={iname}"
        print(f"\n{'='*60}", flush=True)
        print(f"  {name}", flush=True)
        print(f"{'='*60}", flush=True)
        r = evaluate_variant(model, tokenizer, test_data, device, instruction, gen_config, name=name)
        results.append(r)
        print(f"  Acc={r['accuracy']:.2%}, MacroF1={r['macro_f1']:.4f}, "
              f"H-Recall={r['healthy_recall']:.2%}, D-Recall={r['disease_recall']:.2%}",
              flush=True)

    # Summary
    print(f"\n\n{'='*80}", flush=True)
    print(f"  Prompt 调优汇总", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Instruction':<18} {'Acc':<8} {'MacroF1':<10} {'H-Recall':<10} {'D-Recall':<10} {'H-F1':<8} {'D-F1':<8}")
    print(f"{'-'*18} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in sorted(results, key=lambda x: x['disease_recall'], reverse=True):
        print(f"{r['name'].replace('instruct=',''):<18} "
              f"{r['accuracy']:<8.2%} {r['macro_f1']:<10.4f} "
              f"{r['healthy_recall']:<10.2%} {r['disease_recall']:<10.2%} "
              f"{r['healthy_f1']:<8.4f} {r['disease_f1']:<8.4f}")

    with open(os.path.join(OUTPUT_DIR, 'prompt_tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/prompt_tuning_results.json", flush=True)

if __name__ == "__main__":
    main()
