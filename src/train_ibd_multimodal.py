import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
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
print("🚀 IBD 多模态模型训练流程")
print("="*60)

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
trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
total = sum(p.numel() for p in llm.parameters())
print(f"   ✅ 可训练参数：{trainable:,} ({trainable/total*100:.2f}%)")

# 3. 微生物编码器（根据实际数据维度）
print("\n3️⃣ 创建微生物编码器...")
dataset_temp = pd.read_csv("/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv", sep='\t', index_col=0)
num_species = dataset_temp.shape[1]
print(f"   📊 实际 OTU 数量：{num_species}")

microbiome_encoder = MicrobiomeEncoder(num_species=num_species, hidden_size=768).cuda()
projection = nn.Linear(768, 4096).cuda()

# 4. 创建数据集
print("\n4️⃣ 创建 IBD 数据集...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = IBDDataset(
    "/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv",
    "/hd/liujx/microbiome_llm_project/data/ibd_metadata.txt",
    tokenizer
)
print(f"   ✅ 数据集大小：{len(dataset)} 样本")

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(dataloader))
print(f"   ✅ Batch 形状：counts={batch['count_vector'].shape}, text={batch['input_ids'].shape}")

# 5. 测试前向传播
print("\n5️⃣ 测试前向传播...")
counts = batch['count_vector'].cuda()
input_ids = batch['input_ids'].cuda()
attention_mask = batch['attention_mask'].cuda()

with torch.no_grad():
    embeddings = microbiome_encoder(counts)
    projected = projection(embeddings)
    print(f"   微生物编码：{counts.shape} → {embeddings.shape} → {projected.shape}")
    
    outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
    print(f"   LLM 输出：{outputs.logits.shape}")

print("\n✅ IBD 多模态训练流程准备就绪！")
print("="*60)
