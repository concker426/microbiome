#!/usr/bin/env python3
"""
二分类流水线：Healthy vs Disease（IBD+CD+UC）
- 类别平衡：对 Healthy 过采样
- LoRA 微调 Qwen2-0.5B-Instruct
- 测试集评估
"""
import os, re, json, random, copy
import numpy as np
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAIN_DATA = "/hd/liujx/microbiome_llm_project/data/train_set.jsonl"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_binary"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b_binary"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# 训练参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 8
GRAD_ACCUM = 4
EPOCHS = 5
LR = 3e-4
MAX_LENGTH = 1024

BINARY_MAP = {'IBD': 'Disease', 'CD': 'Disease', 'UC': 'Disease', 'Healthy': 'Healthy'}

# ============================================================
# 数据加载与二分类映射
# ============================================================
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def to_binary(item):
    """将样本转为二分类（Healthy / Disease），更新 label 和 assistant 回复"""
    item = copy.deepcopy(item)
    orig = item['label']
    new_label = BINARY_MAP[orig]
    item['label'] = new_label

    # 更新 assistant 回复中的诊断结果
    assistant = item['messages'][1]['content']
    item['messages'][1]['content'] = re.sub(
        r'诊断结果[：:]\s*\w+', f'诊断结果：{new_label}', assistant
    )
    return item

def oversample_healthy(data):
    """对 Healthy 类过采样，使 Healthy 数量 ≈ Disease 数量"""
    healthy = [d for d in data if d['label'] == 'Healthy']
    disease = [d for d in data if d['label'] == 'Disease']
    print(f"  Oversampling: Healthy={len(healthy)}, Disease={len(disease)}")

    # 复制 Healthy 样本直到接近 Disease 数量
    times = len(disease) // len(healthy)
    remainder = len(disease) % len(healthy)
    oversampled = healthy * times + random.sample(healthy, remainder)
    print(f"  Healthy after oversampling: {len(oversampled)}")

    return disease + oversampled

def extract_label(text: str) -> Optional[str]:
    """从模型生成文本中提取二分类标签"""
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        if label in ['Healthy', 'Disease']:
            return label
    for kw in ['Healthy', 'Disease']:
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
        user_len = len(user_ids)
        if user_len > self.max_length - 5:
            user_len = self.max_length - 5

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
# 评估函数
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
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
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
            print(f"  Progress: {idx+1}/{len(test_data)}")

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    all_labels = ['Healthy', 'Disease']
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
        for j in range(len(all_labels)):
            row += f" {cm[i][j]:>10}"
        print(row)

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'predictions': predictions, 'report': report}

# ============================================================
# 主流水线
# ============================================================
def main():
    print("=" * 60)
    print("  二分类流水线: Healthy vs Disease（IBD+CD+UC）")
    print("  基座模型: Qwen2-0.5B-Instruct")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 1. 加载并转换数据
    print(f"\n{'='*60}")
    print(f"  [1/5] 加载并转换数据 → 二分类 + 过采样")
    print(f"{'='*60}")
    raw_train = load_jsonl(TRAIN_DATA)
    raw_test = load_jsonl(TEST_DATA)
    print(f"  原始训练集: {len(raw_train)} 样本")
    print(f"    分布: {dict(Counter(d['label'] for d in raw_train))}")

    train_data = [to_binary(d) for d in raw_train]
    test_data = [to_binary(d) for d in raw_test]
    print(f"  二分类训练集: {dict(Counter(d['label'] for d in train_data))}")
    print(f"  二分类测试集: {dict(Counter(d['label'] for d in test_data))}")

    train_data = oversample_healthy(train_data)
    random.shuffle(train_data)
    print(f"  平衡后训练集: {dict(Counter(d['label'] for d in train_data))}")

    # 2. 加载模型
    print(f"\n{'='*60}")
    print(f"  [2/5] 加载基座模型")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    print(f"  参数量: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M")

    # 3. 零样本评估（基座模型）
    print(f"\n{'='*60}")
    print(f"  [3/5] 零样本评估 (基座模型)")
    print(f"{'='*60}")
    results_before = evaluate(base_model, tokenizer, test_data, device,
                              name="基座模型 (零样本, 二分类)", max_new_tokens=64)

    # 4. LoRA 微调
    print(f"\n{'='*60}")
    print(f"  [4/5] LoRA 微调")
    print(f"{'='*60}")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    train_dataset = QADataset(train_data, tokenizer, max_length=MAX_LENGTH)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=10,
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

    results_after = evaluate(eval_model, tokenizer, test_data, device,
                             name="微调后模型 (二分类)", max_new_tokens=64)

    # 6. 对比报告
    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'基座模型':<22} {'LoRA微调后':<22} {'提升':<22}")
    print(f"{'-'*20} {'-'*22} {'-'*22} {'-'*22}")

    def fmt_change(b, a):
        diff = a - b
        return f"+{diff:.2%}" if diff >= 0 else f"{diff:.2%}"

    acc_before = results_before['accuracy']
    acc_after = results_after['accuracy']
    f1_before = results_before['macro_f1']
    f1_after = results_after['macro_f1']

    print(f"{'准确率':<20} {acc_before:<22.2%} {acc_after:<22.2%} {fmt_change(acc_before, acc_after):<22}")
    print(f"{'Macro F1':<20} {f1_before:<22.4f} {f1_after:<22.4f} {fmt_change(f1_before, f1_after):<22}")

    # 保存结果
    results = {
        'model': 'Qwen2-0.5B-Instruct',
        'task': 'binary_healthy_vs_disease',
        'train_samples_before_balancing': dict(Counter(d['label'] for d in [to_binary(d) for d in raw_train])),
        'train_samples_after_balancing': dict(Counter(d['label'] for d in train_data)),
        'test_samples': dict(Counter(d['label'] for d in test_data)),
        'training_loss': float(final_loss),
        'results_before': {'accuracy': results_before['accuracy'], 'macro_f1': results_before['macro_f1'], 'weighted_f1': results_before['weighted_f1']},
        'results_after': {'accuracy': results_after['accuracy'], 'macro_f1': results_after['macro_f1'], 'weighted_f1': results_after['weighted_f1']},
    }
    with open(os.path.join(EVAL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_before.json'), 'w') as f:
        json.dump(results_before['predictions'], f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_after.json'), 'w') as f:
        json.dump(results_after['predictions'], f, indent=2, ensure_ascii=False)

    print(f"\n✅ 二分类流水线完成！结果保存在 {EVAL_DIR}/")

if __name__ == "__main__":
    main()
