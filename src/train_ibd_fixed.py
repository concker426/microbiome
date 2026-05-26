import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

sys.path.insert(0, "/hd/liujx/microbiome_llm_project")
from src.microbiome_encoder import MicrobiomeEncoder

COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/ibd_fixed_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("🔬 修复版训练脚本 (v2 - 仅保存 LoRA)")
print("="*60)

# 1. 加载数据
print("\n1️⃣ 加载数据...")
counts_df = pd.read_csv(COUNTS_FILE, sep='\t', comment='#', index_col=0)
X = counts_df.values.T.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
col_mean, col_std = X.mean(axis=0), X.std(axis=0)
col_std[col_std == 0] = 1.0
X = (X - col_mean) / (col_std + 1e-8)
num_samples, num_species = X.shape
print(f"   ✅ 样本数: {num_samples}, 物种数: {num_species}")

# 2. 数据集
class IBDDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        label = "disease" if idx % 2 == 0 else "healthy"
        text = f"Microbiome sample. Condition: {label}"
        enc = self.tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
            "microbiome": self.data[idx]
        }

# 3. 加载模型
print("\n2️⃣ 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=quant_config, device_map={"": "cuda:0"}, trust_remote_code=True)

lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none")
llm = get_peft_model(llm, lora_config)
llm.print_trainable_parameters()

# 4. 微生物编码器
micro_encoder = MicrobiomeEncoder(num_species=num_species, hidden_size=768).cuda()
projection = nn.Linear(768, llm.config.hidden_size).cuda()

# 5. 优化器
optimizer = torch.optim.AdamW([
    {'params': llm.parameters(), 'lr': 2e-4},
    {'params': micro_encoder.parameters(), 'lr': 1e-4},
    {'params': projection.parameters(), 'lr': 1e-4}
])

# 6. 训练
print("\n3️⃣ 开始训练...")
dataset = IBDDataset(X, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

llm.train(); micro_encoder.train(); projection.train()
num_epochs, global_step = 3, 0

for epoch in range(num_epochs):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        optimizer.zero_grad()
        outputs = llm(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(), labels=batch["labels"].cuda())
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        global_step += 1
        
        # ✅ 关键修复：使用 save_pretrained 仅保存 LoRA
        if global_step % 100 == 0:
            save_dir = os.path.join(OUTPUT_DIR, f"step_{global_step}")
            llm.save_pretrained(save_dir)  # 只保存 LoRA 权重
            torch.save({
                'micro_encoder': micro_encoder.state_dict(),
                'projection': projection.state_dict()
            }, os.path.join(save_dir, "custom_layers.pt"))
            print(f"\n   💾 保存 Checkpoint (约 50-100MB): {save_dir}")

    print(f"\n✅ Epoch {epoch+1} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

# 7. 保存最终模型
final_dir = os.path.join(OUTPUT_DIR, "final")
llm.save_pretrained(final_dir)
torch.save({'micro_encoder': micro_encoder.state_dict(), 'projection': projection.state_dict()}, os.path.join(final_dir, "custom_layers.pt"))
print(f"\n✅ 完成！模型保存到: {OUTPUT_DIR}")
print(f"   请验证文件大小应该在 50-100MB 左右！")
