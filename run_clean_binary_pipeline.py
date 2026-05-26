#!/usr/bin/env python3
"""
清洗流水线：仅使用短 OTU ID 格式数据（去除格式混淆）
二分类 Healthy vs Disease + 过采样平衡
"""
import os, re, json, random, copy
from collections import Counter
from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAIN_DATA = "/hd/liujx/microbiome_llm_project/data/train_set.jsonl"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_clean_binary"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 8
GRAD_ACCUM = 4
EPOCHS = 5
LR = 3e-4
MAX_LENGTH = 1024

ALL_LABELS_BINARY = ['Healthy', 'Disease']
BINARY_MAP = {'IBD': 'Disease', 'CD': 'Disease', 'UC': 'Disease', 'Healthy': 'Healthy'}

# ============================================================
# Short OTU ID filter
# ============================================================
def is_short_otu_format(content):
    """Check if the OTU data uses short OTU_xxx IDs (not full 16S sequences)"""
    return bool(re.search(r'OTU_\d{4}', content))

def is_long_sequence_format(content):
    """Check if the OTU data uses full 16S rRNA sequences"""
    return bool(re.search(r'[ATCG]{20,}', content))

def filter_short_otu_data(data):
    """Keep only samples with short OTU ID format"""
    filtered = []
    removed = 0
    for d in data:
        content = d['messages'][0]['content']
        if is_short_otu_format(content):
            filtered.append(d)
        else:
            removed += 1
    print(f"  短 OTU 格式: {len(filtered)}, 移除(长序列): {removed}")
    return filtered

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def balance_classes(data):
    label_counts = Counter(d['label'] for d in data)
    max_count = max(label_counts.values())
    print(f"  原始分布: {dict(label_counts)}")
    print(f"  最多类别: {max_count}")
    grouped = {}
    for d in data:
        grouped.setdefault(d['label'], []).append(d)
    balanced = []
    for label, samples in grouped.items():
        if len(samples) == max_count:
            balanced.extend(samples)
        else:
            times = max_count // len(samples)
            remainder = max_count % len(samples)
            oversampled = samples * times + random.sample(samples, remainder)
            balanced.extend(oversampled)
            print(f"  {label}: {len(samples)} → {len(oversampled)}")
    return balanced

def extract_label(text: str) -> Optional[str]:
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in ALL_LABELS_BINARY:
            return label
    for kw in ALL_LABELS_BINARY:
        if kw in text:
            return kw
    cn_map = {'健康': 'Healthy', '疾病': 'Disease'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None

# ============================================================
# Dataset
# ============================================================
def tokenize_chat(tokenizer, messages, max_length=1024, add_generation_prompt=False):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=add_generation_prompt,
        max_length=max_length, truncation=True,
    )
    return encoded.input_ids if hasattr(encoded, 'input_ids') else encoded

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        full_ids = tokenize_chat(self.tokenizer, messages, self.max_length, add_generation_prompt=False)
        user_ids = tokenize_chat(self.tokenizer, [messages[0]], self.max_length, add_generation_prompt=True)
        user_len = min(len(user_ids), self.max_length - 5)
        input_ids = full_ids[:self.max_length]
        labels = [-100] * len(input_ids)
        for i in range(user_len, len(input_ids)):
            labels[i] = input_ids[i]
        return {'input_ids': input_ids, 'attention_mask': [1]*len(input_ids), 'labels': labels}

class DataCollatorForQATraining:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for i in range(len(input_ids)):
            ids = input_ids[i]
            mask = attention_mask[i]
            lbl = labels[i]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_input_ids.append(ids + [self.pad_token_id]*pad_len)
                padded_attention_mask.append(mask + [0]*pad_len)
                padded_labels.append(lbl + [-100]*pad_len)
            else:
                padded_input_ids.append(ids[:max_len])
                padded_attention_mask.append(mask[:max_len])
                padded_labels.append(lbl[:max_len])
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }

# ============================================================
# Evaluation
# ============================================================
def evaluate(model, tokenizer, test_data, device, name="model", max_new_tokens=64):
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []
    for idx, item in enumerate(test_data):
        messages = item['messages']
        true_label = item['label']
        prompt = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
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
        if (idx+1) % 25 == 0:
            print(f"  {idx+1}/{len(test_data)}", flush=True)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS_BINARY, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS_BINARY)
    macro_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS_BINARY, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS_BINARY, average='weighted', zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    header = f"{'':>12}"
    for l in ALL_LABELS_BINARY:
        header += f" {l:>10}"
    print(header)
    for i, label in enumerate(ALL_LABELS_BINARY):
        row = f"{label:>10}:"
        for j in range(2):
            row += f" {cm[i][j]:>10}"
        print(row)
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'predictions': predictions, 'report': report}

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  清洗流水线: 仅短 OTU ID 数据")
    print("  二分类 Healthy vs Disease + 过采样")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 1. Load & filter
    print(f"\n{'='*60}")
    print(f"  [1/5] 加载 + 过滤(仅短 OTU ID) + 二分类 + 过采样")
    print(f"{'='*60}")
    raw_train = filter_short_otu_data(load_jsonl(TRAIN_DATA))
    raw_test = filter_short_otu_data(load_jsonl(TEST_DATA))

    for d in raw_train:
        d['label'] = BINARY_MAP[d['label']]
    for d in raw_test:
        d['label'] = BINARY_MAP[d['label']]

    print(f"  二分类训练集: {dict(Counter(d['label'] for d in raw_train))}")
    print(f"  二分类测试集: {dict(Counter(d['label'] for d in raw_test))}")

    train_data = balance_classes(raw_train)
    random.shuffle(train_data)
    print(f"  平衡后: {len(train_data)} 样本, {dict(Counter(d['label'] for d in train_data))}")

    # 2. Load model
    print(f"\n{'='*60}")
    print(f"  [2/5] 加载 Qwen2-0.5B")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    base_model.config.use_cache = False
    print(f"  参数量: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M")

    # 3. Zero-shot eval
    print(f"\n{'='*60}")
    print(f"  [3/5] 零样本评估")
    print(f"{'='*60}")
    results_before = evaluate(base_model, tokenizer, raw_test, device,
                              name="基座模型 (零样本, 清洗数据)", max_new_tokens=64)

    # 4. LoRA training
    print(f"\n{'='*60}")
    print(f"  [4/5] LoRA 微调")
    print(f"{'='*60}")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM)
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    train_dataset = QADataset(train_data, tokenizer, max_length=MAX_LENGTH)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS,
        learning_rate=LR, bf16=True, logging_steps=10, save_strategy="no",
        remove_unused_columns=False, dataloader_num_workers=0,
        ddp_find_unused_parameters=False, optim="adamw_torch",
        lr_scheduler_type="cosine", warmup_ratio=0.1, report_to="none")
    data_collator = DataCollatorForQATraining(tokenizer, max_length=MAX_LENGTH)
    trainer = Trainer(model=lora_model, args=training_args,
                      train_dataset=train_dataset, data_collator=data_collator)

    print("\n开始训练...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  训练完成！Loss: {final_loss:.4f}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. Post-training eval
    print(f"\n{'='*60}")
    print(f"  [5/5] 微调后评估")
    print(f"{'='*60}")
    del lora_model, trainer
    torch.cuda.empty_cache()

    eval_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map={"": "cuda:0"}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    eval_model = PeftModel.from_pretrained(eval_model, OUTPUT_DIR)
    eval_model.config.use_cache = True

    results_after = evaluate(eval_model, tokenizer, raw_test, device,
                             name="微调后 (仅短 OTU ID, 平衡)", max_new_tokens=64)

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'基座模型':<22} {'LoRA微调后':<22}")
    print(f"{'-'*20} {'-'*22} {'-'*22}")
    for k in ['accuracy', 'macro_f1', 'weighted_f1']:
        print(f"{k:<20} {results_before[k]:<22.4f} {results_after[k]:<22.4f}")

    results = {
        'model': 'Qwen2-0.5B-Instruct',
        'task': 'clean_binary_only_short_otu',
        'train_dist': dict(Counter(d['label'] for d in train_data)),
        'test_dist': dict(Counter(d['label'] for d in raw_test)),
        'training_loss': float(final_loss),
        'results_before': {k: results_before[k] for k in ['accuracy','macro_f1','weighted_f1']},
        'results_after': {k: results_after[k] for k in ['accuracy','macro_f1','weighted_f1']},
    }
    with open(os.path.join(EVAL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(EVAL_DIR, 'predictions_before.json'), 'w') as f:
        json.dump(results_before['predictions'], f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_after.json'), 'w') as f:
        json.dump(results_after['predictions'], f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成！结果: {EVAL_DIR}/")

if __name__ == "__main__":
    main()
