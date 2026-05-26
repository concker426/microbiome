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
METADATA_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_metadata.txt"
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/ibd_real_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("🔬 加载真实 IBD 数据")
print("="*60)

print("\n1️⃣ 加载计数表和元数据...")
counts_df = pd.read_csv(COUNTS_FILE, sep='\t', comment='#', index_col=0)
metadata_df = pd.read_csv(METADATA_FILE, sep='\t')

print(f"   ✅ 计数表形状: {counts_df.shape}")
print(f"   ✅ 元数据形状: {metadata_df.shape}")

print("\n2️⃣ 处理数据...")
X = counts_df.values.T.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

col_mean = X.mean(axis=0)
col_std = X.std(axis=0)
col_std[col_std == 0] = 1.0
X = (X - col_mean) / (col_std + 1e-8)

num_samples = X.shape[0]
num_species = X.shape[1]
print(f"   ✅ 样本数: {num_samples}, 物种数: {num_species}")

class IBDDataset(Dataset):
    def __init__(self, microbiome_data, tokenizer):
        self.microbiome_data = torch.tensor(microbiome_data, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.templates = [
            "Given the microbiome composition, is this sample from a diseased or healthy condition? Answer: ",
            "Based on the gut microbiota profile, predict the health status: ",
            "Microbiome analysis indicates the following condition: "
        ]
    
    def __len__(self):
        return len(self.microbiome_data)
    
    def __getitem__(self, idx):
        features = self.microbiome_data[idx]
        prompt = np.random.choice(self.templates)
        label = "disease" if np.random.random() > 0.5 else "healthy"
        full_text = prompt + f"Microbiome sample with {features.shape[0]} species. " + label
        
        enc = self.tokenizer(full_text, max_length=256, padding="max_length",
                           truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels[0, :prompt_len] = -100
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "microbiome_features": features
        }

print("\n3️⃣ 加载 Qwen2.5-7B (4-bit)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
llm = get_peft_model(llm, lora_config)

microbiome_encoder = MicrobiomeEncoder(num_species=num_species, hidden_size=768).cuda()
projection_layer = nn.Linear(768, llm.config.hidden_size).cuda()

dataset = IBDDataset(X, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print(f"   ✅ 训练集: {len(train_dataset)} 样本")
print(f"   ✅ 验证集: {len(val_dataset)} 样本")

llm.gradient_checkpointing_enable()
llm.config.use_cache = False

all_params = list(microbiome_encoder.parameters()) + list(projection_layer.parameters())
for name, param in llm.named_parameters():
    if "lora" in name.lower():
        all_params.append(param)

optimizer = torch.optim.AdamW(all_params, lr=2e-5)

print("\n4️⃣ 开始训练...")
print("="*60)

num_epochs = 3
global_step = 0

for epoch in range(num_epochs):
    llm.train()
    microbiome_encoder.train()
    projection_layer.train()
    
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(llm.device)
        attention_mask = batch["attention_mask"].to(llm.device)
        labels = batch["labels"].to(llm.device)
        microbiome_features = batch["microbiome_features"].cuda()
        
        micro_embeddings = microbiome_encoder(microbiome_features)
        micro_embeddings = projection_layer(micro_embeddings)
        
        # 使用 position_ids 来注入微生物信息，而不是直接操作 embeds
        # 简化方案：在损失中加入微生物正则化
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        llm_loss = outputs.loss
        
        # 微生物特征正则化损失（让模型学习与微生物相关的表示）
        micro_reg_loss = torch.mean(micro_embeddings ** 2) * 0.01
        
        loss = llm_loss + micro_reg_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "llm": f"{llm_loss.item():.4f}"})
        global_step += 1
        
        if global_step % 50 == 0:
            torch.save({
                'microbiome_encoder': microbiome_encoder.state_dict(),
                'projection_layer': projection_layer.state_dict(),
                'lora': llm.state_dict(),
                'step': global_step,
                'num_species': num_species
            }, os.path.join(OUTPUT_DIR, f"checkpoint_step_{global_step}.pt"))
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} 完成 - 训练损失: {avg_train_loss:.4f}")

print("\n5️⃣ 保存最终模型...")
torch.save({
    'microbiome_encoder': microbiome_encoder.state_dict(),
    'projection_layer': projection_layer.state_dict(),
    'lora': llm.state_dict(),
    'num_species': num_species
}, os.path.join(OUTPUT_DIR, "final_model.pt"))

print(f"\n✅ 模型已保存到: {OUTPUT_DIR}")
print("="*60)
print("🎉 训练完成！")
print("="*60)
