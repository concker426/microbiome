# ProCyon 架构设计演进

## May 2026 — 选 Attention Pool

### 为什么

遵循 MGM 论文：next-genus prediction 预训练 + attention pooling 作为 encoder backbone。这是当时文献里唯一的 microbiome foundation model 方案。

两条假设：

1. Next-genus pretraining 能在 768-dim 向量里编码群落结构
2. Attention pooling 能学会"该关注哪些属"

我们没有质疑这两条假设，因为：
- MGM 论文报告了 "significant improvements"
- ProCyon 论文验证了投影到 LLM 的可行性

### 架构

```
Genus IDs (86 tokens)
  → Embedding(1226,768) + Positional Encoding
  → 6× Transformer Block (causal, 8 heads, FFN=2048)
  → Attention Pooling (learnable query → tanh → softmax)
  → Linear(768→3586)  ← 初版投影
  → concat to Qwen text embeddings
```

### 隐含假设

| 假设 | 当时认为是 | 现在看 |
|------|-----------|--------|
| Attention Pool 能保留属级信息 | True | **False** |
| Transformer 对分类有增益 | True | **待验证** |
| 投影维度足够 | True | 没测过 |

---

## June 2026 — 发现模态坍塌

### 为什么

训练 ProCyon 后评估 Enc+NL vs NL-only 发现：

- dropout=0 时 gap=33.5%：LLM 主要看文本，忽略 microbiome 信号

不是 encoder 的问题，是 LLM 的问题：Qwen 在文本预训练中已经有很强的语言先验，4 个投影 token 难以打破这个先验。

### 应对

引入 modality dropout（随机 zero projection tokens），强制 LLM 在没有 encoder 信号时必须依赖文本：

```
dropout_probs: [0, 0.3, 0.5, 0.7, 0.9]
gap:           33.5% → 5.4%
```

### 发现

dropout=0.5 是最优平衡点：Enc+NL=88.0%，NL-only=79.6%，gap=8.4%。

但这个 gap 仍然存在：NL-only 在 dropout=0.5 时只有 79.6%——说明 LLM 单靠文本不能做得很好，encoder 有 8.4% 的净贡献。

---

## July 2026 — 质疑 MGM

### 转折点

在消融中做了 H1.2b：把 MGM encoder 取出，用纯 MLP 做分类，对比 pretrained vs random。

意外发现了 Baseline C：一个最简单的 encoder——

```python
SimpleEmb = nn.Embedding(1226, 768) + Mean Pool
```

——没有任何训练，准确率 **91.3%**，超过 MGM pretrained + MLP 的 88.6%。

### 这意味着什么

MGM 论文的架构选择——用 attention pooling 把 86 个属压成 1 个向量——是信息瓶颈。简单 mean pooling 保留了更多判别信息。

### 因此

决定系统性回答两个问题：

1. 是不是 attention pooling 导致 MGM 掉点？ → Phase A1
2. 6 层 Transformer 到底有没有贡献？ → Phase A2

---

## July 15 — A1 确认 Attention Pool 是瓶颈

### 控制变量

同一个 SimpleEmb(1226,768)，只变 pooling 方法：

```
SimpleEmb
  ├── Mean Pool   → MLP  → 90.96%
  ├── Attention   → MLP  → 88.06%
  └── CLS Token   → MLP  → 60.93%
```

**Mean pool 比 attention pool 高 2.9 个百分点。**

这直接回答了 5 月埋下的问题："Is attention pooling good enough?"

答案：**No。**

至于 CLS token 的 60.9%——在没有 Transformer 的情况下，CLS 位置没有学到任何东西。这是预期的。

---

## July 15 — A2 确认 Transformer 也是多余的

### 控制变量

同一个 Embedding(1226,768) + Mean Pool，只变是否经过 6 层 Transformer：

```
Emb + Mean                  → MLP → 90.92%
Emb + 6L Transformer + Mean → MLP → 89.43%
```

**Delta: -1.49%**

Transformer 不仅没有帮助，反而降低了性能。

### 证据链

```
Embedding + Mean Pool        → 90.96%
    ↓ 加 6L Transformer
Embedding + Transformer + Mean → 89.43%  (-1.5%)
    ↓ 换 Attention Pool
Embedding + Transformer + Attn → 88.06%  (-2.9%)
```

**MGM 的两个关键设计（Transformer + Attention Pool）在你任务上都造成性能下降，合计损失 ~3%。**

### 为什么

菌属按丰度排序的 ID 序列不是自然语言：
- 没有语法依赖
- token 间关系简单（"出现 + 丰度"比"和谁相邻"更重要）
- Transformer 的优势（学习复杂上下文依赖）在这里变成了噪声

这和 tabular Transformer 文献的发现一致：简单 MLP/Embedding 在表格数据上往往比 Transformer 更稳。

### 研究方向变化

| 之前 | 现在 |
|------|------|
| 如何在 MGM 上做增量改进 | 微生物组分类是否真的需要 Transformer？ |
| ProCyon = MGM + LLM | ProCyon v2 = 轻量 Embedding + Mean Pool + LLM |

---

## July 15 — B1 新架构 ProCyon v2

### 为什么

A1 确认了 attention pooling 是瓶颈。但 SimpleEmb+MLP 是纯分类器，失去了 LLM 的能力（自然语言解释、多任务、交互）。

目标是：**保留 LLM，但用 SimpleEmb 替换 MGM。**

### 新架构

```
Genus IDs (86 tokens)
  → SimpleEmb(1226,768)
  → Mean Pool → 768-dim
  → Projection (LN → 768→7168 → GELU → 7168→14336 → LN → ×0.1)
  → 4×3584 tokens
  → concat to Qwen2.5-7B LoRA text embeddings
  → "Healthy"/"Disease"
```

### 与 MGM 版对比

| | MGM (旧) | ProCyon v2 (新) |
|------|------|------|
| Embedding | 1226×768 | 相同 |
| Transformer | 6 层 causal | **无** |
| Pooling | Attention (learnable) | **Mean** |
| 参数 | 15M | **0.9M** |
| 预训练 | 10k-250k 步 | **无**（随机初始化） |
| Projection | 相同 | 相同 |
| LLM | 相同 | 相同 |

### 预期

- 如果 B1 > MGM+LLM (88.0%)：SimpleEmb 作为新 encoder backbone 成立
- 如果 B1 ≤ MGM+LLM：LLM 没有利用好更好的特征，瓶颈在 projection 或训练策略

---

## July 15 — B2 诊断：Projection 是 LLM 集成的关键

### 为什么 B1 失败了

SimpleEmb + LLM 直接崩到 55.7%，但 SimpleEmb + MLP 是 91.5%。问题不在 representation，在 **representation → LLM 的映射**。

### B2 逐一诊断

| 变体 | Enc+NL | NL-only | Gap | 说明 |
|------|------|------|------|------|
| B1 (SimpleEmb+原Proj) | 55.7% | 58.1% | -2.4% | 基线失败 |
| B2-zero (MGM Proj) | 84.4% | 75.5% | 8.9% | 证明问题在 Projection |
| B2a (LN+Linear) | 58.7% | 56.3% | 2.4% | 线性映射不够 |
| B2b (Adapter 4t) | 85.6% | 71.3% | 14.4% | 超过 MGM Proj |
| **B2c (Adapter 8t)** | **91.6%** | 69.5% | 22.2% | **超过 SimpleEmb+MLP** |

### 核心发现

1. **LLM 可以利用 SimpleEmb**——只要 Projection 正确（B2-zero 恢复 84.4%）
2. **8 tokens > 4 tokens**：4 tokens 是信息瓶颈，8 tokens 释放 +6.0%
3. **Nonlinear adapter > linear**：microbiome manifold → language manifold 需要 GELU
4. **SimpleEmb + LLM > SimpleEmb + MLP**（91.6% > 91.5%）：LLM 首次成为正资产
5. **Gap=22.2%**：encoder 信号被充分利用

### 最终架构

```
Genus IDs → SimpleEmb(1226×768) → Mean Pool → 768-dim
  → LN → Linear(768→2048) → GELU → Linear(2048→3584×8)
  → 8 soft tokens → Qwen2.5-7B LoRA → Diagnosis + Explanation
```

更简单（0.9M vs 15M encoder）、更强（91.6% vs 88.6%）、LLM 真正参与推理。

---

## 已解决的问题

1. ~~Transformer 到底有没有用？~~ → **没有。**（A2）
2. ~~Pooling 哪个最好？~~ → **Mean Pool。**（A1）
3. ~~Projection 哪种最好？~~ → **Adapter 8t。**（B2）
4. ~~LLM 能否利用 SimpleEmb？~~ → **能。需要 Adapter + 8 tokens。**（B2）

## 悬而未决的问题

1. **多 seed 稳定性？** — 待验证
2. **解释能力是否可靠？** — B3 待做
3. **多任务能否进一步提升？** — Phase D
