# ProCyon-Microbiome 项目现状与问题

## 项目概述

构建微生物组基础模型：MGM Encoder（6层Transformer）+ Qwen2.5-7B LLM，实现从菌群数据到自然语言诊断的端到端推理。

---

## 一、架构与流程

```
原始BIOM数据 → Genus聚合(~1200属) → 丰度排序 → Token化(整数序列)
                                                    ↓
                                          ┌──────────┴──────────┐
                                          │  MGM Encoder (预训练) │
                                          │  Next-genus prediction │
                                          │  54k或10k样本          │
                                          │  输出: 768维向量       │
                                          └──────────┬──────────┘
                                                     ↓
                                          ┌──────────┴──────────┐
                                          │  Projection (768→3584)│
                                          │  + Qwen2.5-7B (LoRA) │
                                          │  输出: 自然语言诊断   │
                                          └─────────────────────┘
```

## 二、当前已完成

### 模型

| 变体 | 任务 | 状态 | 关键指标 |
|---|---|---|---|
| NL | 疾病诊断 | ✅ | Super-blind 87.0% |
| NL-aug | 增强诊断 | ✅ | Dropout-only版83.5%（未提升） |
| QA | 自由问答 | ✅ | 需任务专属评估 |
| Subtype | IBD亚型（CD/UC） | ✅ | 88.1% |
| Attribution | 菌属归因 | ✅ | Training loss 0.16 |

### 数据

| 事项 | 状态 |
|---|---|
| AGP+FTP 训练数据 | ✅ 7947条 |
| Qiita 50k 预训练数据 | ✅ 下载+tokenize完成 |
| GMMAD2 知识图谱 | ✅ 四个文件已下载 |
| 样本元数据 528字段 | ✅ 已提取 |

### 基础设施

| 事项 | 状态 |
|---|---|
| 统一超盲评估框架 | ✅ |
| 统一fine-tune launcher | ✅ |
| 传统ML baseline | ✅ RF/XGBoost/LR/SVM |
| 磁盘迁移 | ✅ 根目录97%→92% |

---

## 三、当前问题

### 问题1: 元数据加入后性能下降 ⚠️ 关键

**做了什么**: 将165个样本元数据字段（年龄、饮食、生活习惯、疾病史等）作为自然语言上下文注入NL诊断prompt。

**结果**: ACC从87.0%降到81.0%，Disease Recall降到0%（模型全部预测Healthy）。

**根因**:
- 信噪比太低：528字段中真正和诊断相关的只有~40个
- Prompt过长（~500 tokens），模型注意力分散
- 模型学到捷径（"运动频率高→Healthy"），忽略菌群数据

**待解决**: 需要人工筛选关键字段（见下文），控制在30-50个以内重试。

### 问题2: NL-aug没有提升 ⚠️

**做了什么**: 仅用Dropout扰动（随机丢弃1-3个低丰度菌属）生成增强数据。

**结果**: 干净版NL-aug = 83.5%，与基准NL完全一样，没有任何提升。

**可能的根因**:
- Dropout只扰动低丰度尾端，不影响诊断信号（主要在前10-15个高丰度菌属）
- 需要更强的增强策略，但不引入label泄漏

### 问题3: QA和Attribution无法用现有框架评估 ⚠️

**问题**: QA训练的是回答问题，Attribution训练的是列菌属偏离，但超盲测试集只有诊断prompt。硬用诊断prompt测=全预测Disease。

**待解决**: 需要为QA和Attribution各自设计评估方式。

### 问题4: 跨数据集验证为零 ❌ 重要

目前所有结果都来自AGP+FTP一个数据源。没有在TCMA、外部IBD队列等数据上的验证。

### 问题5: 与已发表模型无直接对比 ❌

没有跑MGM开源代码、没有与Waypoint/BiomeGPT对比数字。传统ML baseline已跑但不在同一测试集上。

### 问题6: 预训练数据量仍小 ⚠️

50k vs MGM的263k vs Waypoint的539k。管线已通，扩展到250k只需要时间和网络。

### 问题7: 多个google.txt要求的任务未实现 ❌

| 任务 | 状态 |
|---|---|
| 表型预测（菌群→年龄/BMI/饮食） | ❌ |
| 群落分类（菌群→取样部位） | ❌ 缺多部位数据 |
| 微生物检索（文本→菌排名） | ❌ |
| BinaryQA（疾病+菌→Yes/No） | ❌ |
| 机制解释（菌+病→代谢通路） | ❌ |
| 多类疾病（不只IBD） | ❌ 只有IBD |

---

## 四、元数据字段选择（待你决定）

> 以下是建议的关键字段。请标记你要保留的，返回编号。

### 疾病史（建议全保留）

```
 24 LIVER_DISEASE
 48 MIGRAINE
 68 SIBO
103 ADD_ADHD
110 IBD
117 IBS
125 CANCER
130 DIABETES
131 CDIFF
150 FUNGAL_OVERGROWTH
164 CARDIOVASCULAR_DISEASE
178 ACID_REFLUX
192 DEPRESSION_BIPOLAR_SCHIZOPHRENIA
218 ASD
220 LUNG_DISEASE
253 KIDNEY_DISEASE
268 THYROID
325 SKIN_CONDITION
426 ALZHEIMERS
456 EPILEPSY_OR_SEIZURE_DISORDER
486 AUTOIMMUNE
```

### 肠道（建议全保留）

```
 66 BOWEL_MOVEMENT_FREQUENCY
381 BOWEL_MOVEMENT_QUALITY
```

### 用药/补充剂

```
102 VITAMIN_D_SUPPLEMENT_FREQUENCY
185 FLU_VACCINE_DATE
259 VITAMIN_B_SUPPLEMENT_FREQUENCY
312 PROBIOTIC_FREQUENCY
348 OTHER_SUPPLEMENT_FREQUENCY
441 CONSUME_ANIMAL_PRODUCTS_ABX
462 ANTIBIOTIC_HISTORY
```

### 饮食

```
 28 ARTIFICIAL_SWEETENERS
 41 SALTED_SNACKS_FREQUENCY
 69 FERMENTED_PLANT_FREQUENCY
 70 SUGARY_SWEETS_FREQUENCY
 84 PREPARED_MEALS_FREQUENCY
 85 HOMECOOKED_MEALS_FREQUENCY
104 READY_TO_EAT_MEALS_FREQUENCY
124 DIET_TYPE
193 FROZEN_DESSERT_FREQUENCY
284 GLUTEN
298 MILK_SUBSTITUTE_FREQUENCY
306 SEAFOOD_FREQUENCY
318 OLIVE_OIL
355 FRUIT_FREQUENCY
366 WHOLE_GRAIN_FREQUENCY
377 LACTOSE
384 TYPES_OF_PLANTS
435 SUGAR_SWEETENED_DRINK_FREQUENCY
457 MILK_CHEESE_FREQUENCY
463 HIGH_FAT_RED_MEAT_FREQUENCY
485 RED_MEAT_FREQUENCY
510 VEGETABLE_FREQUENCY
```

### 身体指标

```
 43 AGE_CAT
 55 AGE_YEARS
135 HEIGHT_CM
187 BMI
215 SEX
288 WEIGHT_KG
404 WEIGHT_CHANGE
416 RACE
481 BMI_CAT
487 BIRTH_YEAR
521 AGE_CORRECTED
```

### 生活方式

```
 60 EXERCISE_FREQUENCY
140 LAST_TRAVEL
153 ONE_LITER_OF_WATER_A_DAY_FREQUENCY
165 DRINKS_PER_SESSION
225 ALCOHOL_FREQUENCY
240 SMOKING_FREQUENCY
344 EXERCISE_LOCATION
372 SLEEP_DURATION
492 ALCOHOL_CONSUMPTION
```

### 环境

```
 73 DOG
111 CAT
191 LAST_MOVE
194 LIVINGWITH
209 ROOMMATES
477 DRINKING_WATER_SOURCE
```

### 人口学

```
 12 COUNTRY_OF_BIRTH
177 COUNTRY
250 LEVEL_OF_EDUCATION
360 STATE
371 COUNTRY_RESIDENCE
```

### 手术史

```
  8 TONSILS_REMOVED
415 CSECTION
476 APPENDIX_REMOVED
```

---

## 五、下一步计划

| 优先级 | 事项 | 阻塞因素 |
|---|---|---|
| P0 | 选定元数据字段→重训 | **等你返回编号** |
| P0 | NL+精选元数据 super-blind | 重训完成后 |
| P1 | 跨数据集验证 | 需下载外部数据 |
| P1 | 跑MGM对比 | GitHub网络问题 |
| P2 | QA/Attribution评估 | 需定义评估标准 |
| P2 | 表型预测任务 | 等元数据模型稳定后 |
| P2 | 扩展到250k预训练 | 需网络下载时间 |
