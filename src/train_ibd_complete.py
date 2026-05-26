import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, '/hd/liujx/microbiome_llm_project/src')

from microbiome_encoder import MicrobiomeEncoder

class IBDDataset(Dataset):
    def __init__(self, count_file, metadata_file, tokenizer, max_length=512):
        self.counts = pd.read_csv(count_file, sep='\t', index_col=0)
        self.metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.counts)
    
    def __getitem__(self, idx):
        sample_id = self.counts.index[idx]
        count_vector = torch.tensor(self.counts.iloc[idx].values, dtype=torch.float32)
        
        disease = self.metadata.loc[sample_id, 'Disease']
        description = self.metadata.loc[sample_id, 'Description']
        
        prompt = f"Patient microbiome analysis: {description}. Disease status: {disease}."
        
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'count_vector': count_vector,
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

print("="*60)
print("🚀 IBD 多模态模型 - 完整训练")
print("="*60)

# 配置
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
SAVE_DIR = "/hd/liujx/microbiome_llm_project/checkpoints/ibd_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 加载模型
print("\n1️⃣ 加载 Qwen2.5-7B（4-bit）...")
model_path = "/hd/liujx/microbiome_llm_project/models/qwen2-7b"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# 2. LoRA 配置
print("\n2️⃣ 配置 LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
llm = get_peft_model(llm, lora_config)

# 3. 微生物编码器
print("\n3️⃣ 创建微生物编码器...")
dataset_temp = pd.read_csv("/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv", sep='\t', index_col=0)
num_species = dataset_temp.shape[1]
microbiome_encoder = MicrobiomeEncoder(num_species=num_species, hidden_size=768).cuda()
projection = nn.Linear(768, 4096).cuda()

# 4. 数据集
print("\n4️⃣ 创建数据集...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = IBDDataset(
    "/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv",
    "/hd/liujx/microbiome_llm_project/data/ibd_metadata.txt",
    tokenizer
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"   ✅ 数据集：{len(dataset)} 样本，{len(dataloader)} batches")

# 5. 优化器
print("\n5️⃣ 配置优化器...")
optimizer = torch.optim.AdamW(
    list(llm.parameters()) + list(projection.parameters()) + list(microbiome_encoder.parameters()),
    lr=LEARNING_RATE
)

# 6. 训练循环
print("\n6️⃣ 开始训练...")
global_step = 0
for epoch in range(EPOCHS):
    llm.train()
    microbiome_encoder.train()
    projection.train()
    
    epoch_loss = 0
    for step, batch in enumerate(dataloader):
        counts = batch['count_vector'].cuda()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        
        optimizer.zero_grad()
        
        embeddings = microbiome_encoder(counts)
        projected = projection(embeddings)
        
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        if global_step % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS}, Step {global_step}: Loss = {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")
    
    # 保存 checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 保存 LoRA 权重
    llm.save_pretrained(os.path.join(checkpoint_path, "lora"))
    
    # 保存编码器和投影层
    torch.save({
        'microbiome_encoder': microbiome_encoder.state_dict(),
        'projection': projection.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'loss': avg_loss
    }, os.path.join(checkpoint_path, "encoder_projection.pt"))
    
    print(f"   💾 Checkpoint 保存到: {checkpoint_path}")

print("\n" + "="*60)
print("✅ 训练完成！模型已保存")
print("="*60)
