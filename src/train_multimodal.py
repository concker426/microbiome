import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sys
sys.path.insert(0, '/hd/liujx/microbiome_llm_project/src')

from microbiome_dataset import MicrobiomeDataset
from microbiome_encoder import MicrobiomeEncoder

print("="*50)
print("🚀 微生物多模态模型训练")
print("="*50)

# 1. 加载 Qwen2.5-7B
print("\n1️⃣ 加载 Qwen2.5-7B-Instruct...")
model_path = "/hd/liujx/microbiome_llm_project/models/qwen2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
print(f"   ✅ Qwen2.5-7B 加载成功")

# 2. 配置 LoRA
print("\n2️⃣ 配置 LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
llm = get_peft_model(llm, lora_config)
trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
total = sum(p.numel() for p in llm.parameters())
print(f"   ✅ LoRA 可训练参数：{trainable:,} ({trainable/total*100:.2f}%)")

# 3. 创建微生物编码器
print("\n3️⃣ 创建微生物编码器...")
microbiome_encoder = MicrobiomeEncoder(num_species=500, hidden_size=768)

# 4. 创建投影层（768 → 4096，对齐 Qwen 的 hidden_size）
print("\n4️⃣ 创建投影层...")
projection = nn.Linear(768, 4096)  # Qwen2.5-7B 的 hidden_size 是 4096

# 5. 测试完整流程
print("\n5️⃣ 测试完整流程...")

# 创建示例数据
import numpy as np
fake_count = torch.randn(2, 500)
embedding = microbiome_encoder(fake_count)
print(f"   微生物编码：{fake_count.shape} → {embedding.shape}")

projected = projection(embedding)
print(f"   投影到 LLM 空间：{embedding.shape} → {projected.shape}")

# 测试 LLM 输入
text = "This patient has inflammatory bowel disease."
inputs = tokenizer(text, return_tensors="pt").to(llm.device)
print(f"   文本编码：{inputs.input_ids.shape}")

print("\n✅ 完整流程测试成功！")
print("="*50)
