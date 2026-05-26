import torch
import sys
import os

# 添加 src 目录到路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from microbiome_encoder import MicrobiomeEncoder
from microbiome_dataset import MicrobiomeDataset

# ... existing code ...
import pandas as pd
import numpy as np
import os

print("="*50)
print("🧪 微生物模型测试")
print("="*50)

# 测试编码器
print("\n1. 测试编码器")
encoder = MicrobiomeEncoder(num_species=100, hidden_size=768, num_layers=2)
count_data = torch.randn(16, 100)
embedding = encoder(count_data)
print(f"输入：{count_data.shape}")
print(f"输出：{embedding.shape}")
print("✅ 编码器测试通过")

# 测试数据集
print("\n2. 测试数据集")
temp_path = "/hd/liujx/microbiome_llm_project/data/test.csv"
os.makedirs(os.path.dirname(temp_path), exist_ok=True)

num_samples = 50
num_species = 100
sample_ids = [f"s{i:03d}" for i in range(num_samples)]
species_ids = [f"sp{i:03d}" for i in range(num_species)]
count_data = np.random.poisson(lam=10, size=(num_samples, num_species))

pd.DataFrame(count_data, index=sample_ids, columns=species_ids).to_csv(temp_path)

dataset = MicrobiomeDataset(temp_path)
sample = dataset[0]
print(f"样本形状：{sample['count_vector'].shape}")
print("✅ 数据集测试通过")

os.remove(temp_path)

print("\n" + "="*50)
print("✅ 所有测试通过！")
print("="*50)