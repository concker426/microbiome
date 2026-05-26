# 多数据集联合训练指南

## 📋 问题背景

项目中有两个不同的微生物组数据集：
- **Study数据集**: 6374个样本 × 421个特征（DNA序列ID）
- **IBD数据集**: 500个样本 × 300个特征（sample_xxx格式ID）

之前的问题：
1. 编码器架构不统一（Transformer vs MLP混用）
2. 数据维度不匹配导致无法同时使用两个数据集
3. 模型性能不佳（准确率仅38%）

## ✅ 解决方案

### 1. 统一的Transformer编码器架构

创建了 `unified_microbiome_encoder.py`，包含：
- **UnifiedMicrobiomeEncoder**: Transformer架构的编码器
- 支持动态输入维度（通过 `create_for_dataset()` 方法）
- 为每个数据集创建独立的encoder实例

```python
# Study数据集 (421维)
encoder_study = UnifiedMicrobiomeEncoder.create_for_dataset("study")

# IBD数据集 (300维)
encoder_ibd = UnifiedMicrobiomeEncoder.create_for_dataset("ibd")
```

### 2. 数据合并策略

运行 `merge_datasets.py` 生成统一的训练数据：

```bash
python3 src/merge_datasets.py
```

这会生成：
- `data/merged_training_data.jsonl`: 合并的训练样本
- `data/merged_training_data_labels.json`: 所有样本的标签映射

### 3. 改进的训练流程

使用 `train_merged_multidataset.py` 进行训练：

```bash
python3 src/train_merged_multidataset.py
```

**关键改进：**
- ✅ 使用Transformer编码器（更强的表达能力）
- ✅ 支持两个数据集同时训练
- ✅ 更大的batch size (2) + 梯度累积 (4)
- ✅ 更低的学习率 (1e-4)
- ✅ 更多的训练轮数 (5 epochs)
- ✅ 自动保存最佳模型
- ✅ Weight decay正则化

## 🚀 使用步骤

### 步骤1: 生成合并数据

```bash
cd /hd/liujx/microbiome_llm_project
python3 src/merge_datasets.py
```

### 步骤2: 开始训练

```bash
python3 src/train_merged_multidataset.py
```

训练过程会：
- 每100步保存一个checkpoint
- 每个epoch保存一次模型
- 自动跟踪并保存最佳模型（最低loss）

### 步骤3: 评估模型

训练完成后，可以使用以下脚本进行评估（需要更新推理脚本以支持双encoder）。

## 📊 预期效果

相比之前的训练：
- **更多数据**: 从500样本增加到6874样本（13.7倍）
- **更好架构**: Transformer比MLP有更强的特征提取能力
- **更稳定训练**: 更大的有效batch size和合适的学习率
- **更高准确率**: 预期从38%提升到60-70%+

## 🔧 技术细节

### 编码器结构

```
输入 (421或300维)
    ↓
Linear投影 (→ 768维)
    ↓
Transformer Encoder (2层, 8头注意力)
    ↓
LayerNorm + Dropout
    ↓
输出 (768维embedding)
    ↓
Projection Layer (768 → 3584)
    ↓
Qwen2.5-7B LLM
```

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 2 | 每个GPU批次大小 |
| Gradient Accumulation | 4 | 有效batch size = 8 |
| Learning Rate | 1e-4 | AdamW优化器 |
| Weight Decay | 0.01 | L2正则化 |
| Epochs | 5 | 训练轮数 |
| LoRA Rank | 16 | 低秩适配维度 |
| LoRA Alpha | 32 | LoRA缩放因子 |

## ⚠️ 注意事项

1. **显存需求**: 需要至少24GB GPU显存（使用4bit量化）
2. **训练时间**: 预计6-12小时（取决于GPU性能）
3. **数据质量**: Study数据集的标签是推断的（基于样本ID），可能不够准确
4. **维度隔离**: 两个encoder完全独立，不会共享权重

## 🔄 后续优化方向

1. **标签优化**: 为Study数据集获取真实标签
2. **数据增强**: 对少数类进行过采样
3. **课程学习**: 先训练简单样本，再训练困难样本
4. **集成学习**: 结合多个模型的预测结果
5. **可解释性**: 添加注意力可视化，了解哪些菌种最重要

## 📝 文件清单

- `src/unified_microbiome_encoder.py` - 统一编码器实现
- `src/merge_datasets.py` - 数据合并脚本
- `src/train_merged_multidataset.py` - 改进的训练脚本
- `data/merged_training_data.jsonl` - 生成的训练数据
- `saved_models/merged_multidataset_v1/` - 训练输出的模型

---

**创建时间**: 2026-05-06  
**作者**: AI Assistant  
**版本**: 1.0
