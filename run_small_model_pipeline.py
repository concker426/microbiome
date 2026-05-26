#!/usr/bin/env python3
"""
完整流水线：使用小模型 (Qwen2-0.5B-Instruct) 进行微生物组 QA 任务
1. 加载数据（已划分训练/测试集）
2. 零样本评估基座模型（测试集）
3. LoRA 微调（训练集）
4. 微调后评估（测试集）
5. 对比报告
"""

import os, re, json, sys
import numpy as np
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2-0.5B-Instruct"
TRAIN_DATA = "/hd/liujx/microbiome_llm_project/data/train_set.jsonl"
TEST_DATA = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/qwen2_0.5b_qa"
EVAL_DIR = "/hd/liujx/microbiome_llm_project/eval_results_0.5b"
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

# ============================================================
# 数据加载
# ============================================================
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_label(text: str) -> Optional[str]:
    """从模型生成文本中提取诊断标签"""
    # 精确匹配 "诊断结果：LABEL"
    m = re.search(r'诊断结果[：:]\s*(\S+)', text)
    if m:
        label = m.group(1).strip('。，, \n')
        return label
    # 直接包含标签关键字
    for kw in ['Healthy', 'IBD', 'CD', 'UC']:
        if kw in text:
            return kw
    # 中文匹配
    cn_map = {'健康': 'Healthy', '炎症性肠病': 'IBD', '克罗恩': 'CD', '溃疡性结肠炎': 'UC'}
    for cn, en in cn_map.items():
        if cn in text:
            return en
    return None

# ============================================================
# 数据集
# ============================================================
def tokenize_chat(tokenizer, messages, max_length=1024, add_generation_prompt=False):
    """使用 tokenizer 的 chat template 进行编码，返回 input_ids 列表"""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        max_length=max_length,
        truncation=True,
    )
    # transformers 5.x 返回 BatchEncoding 对象
    if hasattr(encoded, 'input_ids'):
        return encoded.input_ids
    return encoded

def tokenize_chat_str(tokenizer, messages, max_length=1024, add_generation_prompt=False):
    """使用 tokenizer 的 chat template 编码为文本"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']  # [user_msg, assistant_msg]

        # 编码完整对话（user + assistant）
        full_ids = tokenize_chat(self.tokenizer, messages, self.max_length, add_generation_prompt=False)

        # 编码 user 部分（含 generation prompt），用于定位 assistant 开始位置
        user_ids = tokenize_chat(self.tokenizer, [messages[0]], self.max_length, add_generation_prompt=True)
        user_len = len(user_ids)
        if user_len > self.max_length - 5:
            user_len = self.max_length - 5

        input_ids = full_ids[:self.max_length]
        labels = [-100] * len(input_ids)
        # assistant 部分的 token 参与 loss
        for i in range(user_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels,
        }


class DataCollatorForQATraining:
    """为 QA 训练数据做 padding"""
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, self.max_length)

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
    """在测试集上评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    pred_labels = []

    for idx, item in enumerate(test_data):
        messages = item['messages']
        true_label = item['label']

        prompt = tokenize_chat_str(tokenizer, [messages[0]], MAX_LENGTH, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
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
            print(f"    评估进度: {idx+1}/{len(test_data)}")

    # 计算指标
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    all_labels = ['Healthy', 'IBD', 'CD', 'UC']
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=all_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='weighted', zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name} 评估结果")
    print(f"{'='*60}")
    print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\n分类报告:")
    print(report)
    print(f"\n混淆矩阵:")
    header = f"{'':>12}"
    for l in all_labels:
        header += f" {l:>10}"
    print(header)
    for i, label in enumerate(all_labels):
        row = f"{label:>10}:"
        for j in range(4):
            row += f" {cm[i][j]:>10}"
        print(row)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': predictions,
        'report': report,
        'confusion_matrix': cm.tolist(),
    }


# ============================================================
# 主流水线
# ============================================================
def main():
    print("=" * 60)
    print("  微生物组 QA 任务流水线")
    print("  基座模型: Qwen2-0.5B-Instruct")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  [1/5] 加载数据")
    print(f"{'='*60}")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    print(f"  训练集: {len(train_data)} 样本")
    train_labels = Counter(d['label'] for d in train_data)
    print(f"    标签分布: {dict(train_labels)}")
    print(f"  测试集: {len(test_data)} 样本")
    test_labels = Counter(d['label'] for d in test_data)
    print(f"    标签分布: {dict(test_labels)}")

    # ============================================================
    # 2. 加载 Tokenizer 和基座模型
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  [2/5] 加载基座模型 (Qwen2-0.5B-Instruct)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Qwen2-0.5B 很小，可以直接用 BF16 加载（约 1GB），不需要量化
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"  模型参数量: {param_count/1e6:.1f}M")

    if torch.cuda.is_available():
        print(f"  显存占用: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # ============================================================
    # 3. 零样本评估（基座模型）
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  [3/5] 零样本评估 (基座模型, 未训练)")
    print(f"{'='*60}")
    results_before = evaluate(
        base_model, tokenizer, test_data, device,
        name="基座模型 (零样本)", max_new_tokens=64
    )

    if torch.cuda.is_available():
        print(f"  显存占用: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # ============================================================
    # 4. 应用 LoRA 并训练
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  [4/5] LoRA 微调")
    print(f"{'='*60}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
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
        save_strategy="no",  # 不保存中间 checkpoint 以节省时间
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
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n开始训练...")
    train_result = trainer.train()
    final_loss = train_result.training_loss
    print(f"  训练完成！平均 Loss: {final_loss:.4f}")

    # 保存模型
    print("\n保存 LoRA 权重...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if torch.cuda.is_available():
        print(f"  显存占用: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # ============================================================
    # 5. 训练后评估
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  [5/5] 微调后评估")
    print(f"{'='*60}")

    # 卸载训练模型
    del lora_model, trainer
    torch.cuda.empty_cache()

    # 重新加载基座 + LoRA（强制单卡，避免设备不一致）
    eval_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    eval_model = PeftModel.from_pretrained(eval_model, OUTPUT_DIR)
    eval_model.config.use_cache = True

    if torch.cuda.is_available():
        print(f"  显存占用: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    results_after = evaluate(
        eval_model, tokenizer, test_data, device,
        name="微调后模型", max_new_tokens=64
    )

    # ============================================================
    # 6. 对比报告
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'基座模型(零样本)':<22} {'LoRA微调后':<22} {'提升':<22}")
    print(f"{'-'*20} {'-'*22} {'-'*22} {'-'*22}")

    acc_before = results_before['accuracy']
    acc_after = results_after['accuracy']
    f1_before = results_before['macro_f1']
    f1_after = results_after['macro_f1']
    wf1_before = results_before['weighted_f1']
    wf1_after = results_after['weighted_f1']

    def fmt_change(b, a):
        diff = a - b
        if diff >= 0:
            return f"+{diff:.2%}"
        return f"{diff:.2%}"

    print(f"{'准确率':<20} {acc_before:<22.2%} {acc_after:<22.2%} {fmt_change(acc_before, acc_after):<22}")
    print(f"{'Macro F1':<20} {f1_before:<22.4f} {f1_after:<22.4f} {fmt_change(f1_before, f1_after):<22}")
    print(f"{'Weighted F1':<20} {wf1_before:<22.4f} {wf1_after:<22.4f} {fmt_change(wf1_before, wf1_after):<22}")

    # 保存结果
    results = {
        'model': 'Qwen2-0.5B-Instruct',
        'task': 'microbiome_qa',
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'training_loss': float(final_loss),
        'results_before': {
            'accuracy': results_before['accuracy'],
            'macro_f1': results_before['macro_f1'],
            'weighted_f1': results_before['weighted_f1'],
        },
        'results_after': {
            'accuracy': results_after['accuracy'],
            'macro_f1': results_after['macro_f1'],
            'weighted_f1': results_after['weighted_f1'],
        },
        'improvement': {
            'accuracy': acc_after - acc_before,
            'macro_f1': f1_after - f1_before,
            'weighted_f1': wf1_after - wf1_before,
        }
    }
    with open(os.path.join(EVAL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {EVAL_DIR}/")

    # 保存详细预测
    with open(os.path.join(EVAL_DIR, 'predictions_before.json'), 'w') as f:
        json.dump(results_before['predictions'], f, indent=2, ensure_ascii=False)
    with open(os.path.join(EVAL_DIR, 'predictions_after.json'), 'w') as f:
        json.dump(results_after['predictions'], f, indent=2, ensure_ascii=False)

    print("\n✅ 流水线完成！")


if __name__ == "__main__":
    main()
