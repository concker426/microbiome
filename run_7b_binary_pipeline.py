#!/usr/bin/env python3
"""
7B 二分类流水线：Healthy vs Disease（过采样平衡）
- Qwen2.5-7B-Instruct + LoRA
- 纯文本输入（与 0.5B 一致，无自定义编码器）
"""
import os, re, json, random, copy
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
TRAIN_DATA = "/hd/liujx/microbiome_llm_project/data/train_set.jsonl"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/qwen2.5_7b_binary"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_7b_binary"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1          # 7B 模型显存更大，小 batch
GRAD_ACCUM = 8          # 有效 batch = 8
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 1024

BINARY_MAP = {'IBD': 'Disease', 'CD': 'Disease', 'UC': 'Disease', 'Healthy': 'Healthy'}
ALL_LABELS = ['Healthy', 'Disease']

# ============================================================
# 数据加载与过采样
# ============================================================
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
        if label in ALL_LABELS:
            return label
    for kw in ALL_LABELS:
        if kw in text:
            return kw
    cn_map = {'健康': 'Healthy', '疾病': 'Disease'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None

# ============================================================
# 数据集
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

        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels,
        }

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
                padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_attention_mask.append(mask + [0] * pad_len)
                padded_labels.append(lbl + [-100] * pad_len)
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
# 评估
# ============================================================
def evaluate(model, tokenizer, test_data, device, name="model", max_new_tokens=64):
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        messages = item['messages']
        true_label = item['label']
        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
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
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(test_data)}", flush=True)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
    macro_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, labels=ALL_LABELS, average='weighted', zero_division=0)

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
    for l in ALL_LABELS:
        header += f" {l:>10}"
    print(header)
    for i, label in enumerate(ALL_LABELS):
        row = f"{label:>10}:"
        for j in range(len(ALL_LABELS)):
            row += f" {cm[i][j]:>10}"
        print(row)

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'predictions': predictions, 'report': report}

# ============================================================
# 主流水线
# ============================================================
def main():
    print("=" * 60)
    print("  7B 二分类流水线: Healthy vs Disease（过采样平衡）")
    print("  基座模型: Qwen2.5-7B-Instruct")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  显存: {mem/1e9:.1f} GB")

    # 1. 数据加载
    print(f"\n{'='*60}")
    print(f"  [1/5] 加载数据 + 二分类 + 过采样")
    print(f"{'='*60}")
    raw_train = load_jsonl(TRAIN_DATA)
    raw_test = load_jsonl(TEST_DATA)
    print(f"  原始训练集: {len(raw_train)}")

    # 转二分类
    for d in raw_train:
        d['label'] = BINARY_MAP[d['label']]
    for d in raw_test:
        d['label'] = BINARY_MAP[d['label']]

    print(f"  二分类训练集: {dict(Counter(d['label'] for d in raw_train))}")
    print(f"  二分类测试集: {dict(Counter(d['label'] for d in raw_test))}")

    train_data = balance_classes(raw_train)
    random.shuffle(train_data)
    print(f"  平衡后训练集: {len(train_data)} 样本, 分布: {dict(Counter(d['label'] for d in train_data))}")

    # 2. 加载模型（BF16，无需量化）
    print(f"\n{'='*60}")
    print(f"  [2/5] 加载 Qwen2.5-7B（BF16）")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count/1e9:.2f}B")

    if torch.cuda.is_available():
        print(f"  显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 3. 零样本评估
    print(f"\n{'='*60}")
    print(f"  [3/5] 零样本评估（基座模型）")
    print(f"{'='*60}")
    results_before = evaluate(model, tokenizer, raw_test, device,
                              name="基座模型 (零样本, 7B)", max_new_tokens=64)

    # 4. LoRA 微调
    print(f"\n{'='*60}")
    print(f"  [4/5] LoRA 微调")
    print(f"{'='*60}")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    train_dataset = QADataset(train_data, tokenizer, max_length=MAX_LENGTH)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
    )
    data_collator = DataCollatorForQATraining(tokenizer, max_length=MAX_LENGTH)
    trainer = Trainer(
        model=lora_model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
    )

    print("\n开始训练...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  训练完成！平均 Loss: {final_loss:.4f}")

    print("\n保存 LoRA 权重...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. 训练后评估
    print(f"\n{'='*60}")
    print(f"  [5/5] 微调后评估")
    print(f"{'='*60}")
    del lora_model, trainer
    torch.cuda.empty_cache()

    eval_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map={"": "cuda:0"},
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    eval_model = PeftModel.from_pretrained(eval_model, OUTPUT_DIR)
    eval_model.config.use_cache = True

    results_after = evaluate(eval_model, tokenizer, raw_test, device,
                             name="微调后模型 (7B, 二分类, 平衡)", max_new_tokens=64)

    # 6. 对比
    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'基座模型(7B)':<22} {'LoRA微调后':<22} {'提升':<22}")
    print(f"{'-'*20} {'-'*22} {'-'*22} {'-'*22}")

    def fmt_change(b, a):
        diff = a - b
        return f"+{diff:.2%}" if diff >= 0 else f"{diff:.2%}"

    for k in ['accuracy', 'macro_f1', 'weighted_f1']:
        b_val = results_before.get(k, 0)
        a_val = results_after.get(k, 0)
        print(f"{k:<20} {b_val:<22.4f} {a_val:<22.4f} {fmt_change(b_val, a_val)}")

    # 保存结果
    results = {
        'model': 'Qwen2.5-7B-Instruct',
        'task': 'binary_healthy_vs_disease_balanced',
        'train_dist': dict(Counter(d['label'] for d in train_data)),
        'test_dist': dict(Counter(d['label'] for d in raw_test)),
        'training_loss': float(final_loss),
        'results_before': {k: results_before[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
        'results_after': {k: results_after[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
    }
    with open(os.path.join(EVAL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_before.json'), 'w') as f:
        json.dump(results_before['predictions'], f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_after.json'), 'w') as f:
        json.dump(results_after['predictions'], f, indent=2, ensure_ascii=False)

    print(f"\n✅ 7B 二分类流水线完成！结果: {EVAL_DIR}/")

if __name__ == "__main__":
    main()
