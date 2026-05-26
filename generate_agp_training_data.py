"""
American Gut Project 数据 → genus 级自然语言描述 + LLM 训练数据
"""
import pandas as pd
import numpy as np
import json
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_DIR = "/hd/liujx/microbiome_llm_project/data"
TAX_FILE = os.path.join(DATA_DIR, "gut_taxonomic_table.csv")
META_FILE = os.path.join(DATA_DIR, "sample_data.csv")
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/data/agp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N_GENERA = 15  # 每个样本保留 top N 个属
RANDOM_STATE = 42

# === 标签映射 ===
IBD_DIAG_MAP = {
    "Ulcerative colitis": "UC",
    "Colonic Crohn's Disease": "CD",
    "Ileal Crohn's Disease": "CD",
    "Ileal and Colonic Crohn's Disease": "CD",
    "Microcolitis": "UC",
}

# 扩展模式：包含更多样本
INCLUDE_SELF_DIAGNOSED = True        # Self-diagnosed 也算 Disease
INCLUDE_ALTERNATIVE_DX = True        # Alternative medicine 也算
INCLUDE_UNSPECIFIED = True           # Unspecified 也算 Disease（最大增量）


def load_and_link_data():
    """加载 taxonomy 表和 metadata，通过 run_accession 链接"""
    print("加载 taxonomy 表...")
    tax = pd.read_csv(TAX_FILE, low_memory=False)

    print("加载 metadata...")
    meta = pd.read_csv(META_FILE, low_memory=False)

    # taxonomy 的 sample 列格式: PRJEB11419_ERRxxxxx
    tax["run_accession"] = tax["sample"].str.replace("PRJEB11419_", "", regex=False)

    # 合并
    df = tax.merge(meta[["run_accession", "attribute_ibd", "attribute_ibd_diagnosis_refined"]],
                   on="run_accession", how="inner")

    print(f"  链接后样本数: {len(df)}")
    return df


def filter_samples(df):
    """只保留明确健康 + 多种标准扩大的 IBD 样本"""
    healthy = df[df["attribute_ibd"] == "I do not have this condition"].copy()
    healthy["label"] = "Healthy"
    healthy["diagnosis_detail"] = ""
    healthy["label_confidence"] = "high"

    # 确诊 IBD
    dx_medical = df[
        df["attribute_ibd"] == "Diagnosed by a medical professional (doctor, physician assistant)"
    ].copy()
    dx_medical["label"] = dx_medical["attribute_ibd_diagnosis_refined"].map(IBD_DIAG_MAP)
    dx_medical["diagnosis_detail"] = dx_medical["attribute_ibd_diagnosis_refined"]
    dx_medical["label_confidence"] = "high"
    dx_medical = dx_medical[dx_medical["label"].notna()]  # 去掉 Unspecified
    print(f"  确诊 medical: {len(dx_medical)}")

    # Self-diagnosed
    disease_groups = [dx_medical]
    if INCLUDE_SELF_DIAGNOSED:
        self_dx = df[df["attribute_ibd"] == "Self-diagnosed"].copy()
        self_dx["label"] = self_dx["attribute_ibd_diagnosis_refined"].map(IBD_DIAG_MAP)
        self_dx["diagnosis_detail"] = self_dx["attribute_ibd_diagnosis_refined"]
        self_dx["label_confidence"] = "low"
        self_dx = self_dx[self_dx["label"].notna()]
        disease_groups.append(self_dx)
        print(f"  Self-diagnosed: {len(self_dx)}")

    # Alternative medicine
    if INCLUDE_ALTERNATIVE_DX:
        alt_dx = df[df["attribute_ibd"] == "Diagnosed by an alternative medicine practitioner"].copy()
        alt_dx["label"] = alt_dx["attribute_ibd_diagnosis_refined"].map(IBD_DIAG_MAP)
        alt_dx["diagnosis_detail"] = alt_dx["attribute_ibd_diagnosis_refined"]
        alt_dx["label_confidence"] = "low"
        alt_dx = alt_dx[alt_dx["label"].notna()]
        disease_groups.append(alt_dx)
        print(f"  Alternative medicine: {len(alt_dx)}")

    # Unspecified (属性为 Unspecified 但诊断细节中有具体分型)
    if INCLUDE_UNSPECIFIED:
        unspec = df[
            (df["attribute_ibd"] == "Unspecified") &
            (df["attribute_ibd_diagnosis_refined"].isin(IBD_DIAG_MAP.keys()))
        ].copy()
        unspec["label"] = unspec["attribute_ibd_diagnosis_refined"].map(IBD_DIAG_MAP)
        unspec["diagnosis_detail"] = unspec["attribute_ibd_diagnosis_refined"]
        unspec["label_confidence"] = "low"
        disease_groups.append(unspec)
        print(f"  Unspecified+detail: {len(unspec)}")

    # 合并所有 Disease
    diagnosed = pd.concat(disease_groups, ignore_index=True)
    print(f"  IBD 总计: {len(diagnosed)}")

    result = pd.concat([healthy, diagnosed], ignore_index=True)
    print(f"\n过滤后总计: {len(result)} (Healthy: {len(healthy)}, Disease: {len(diagnosed)})")
    return result


def extract_top_genera(row, top_n=TOP_N_GENERA):
    """
    从一行中提取 top N 属及其相对丰度。
    只取 genus 级列（6个 level 的列名）。
    """
    # genus 级列名: Bacteria.phylum.class.order.family.genus
    genus_data = {}
    for col_name, val in row.items():
        if col_name in ["Unnamed: 0", "sample", "run_accession",
                        "attribute_ibd", "attribute_ibd_diagnosis_refined",
                        "label", "diagnosis_detail"]:
            continue
        levels = str(col_name).split(".")
        if len(levels) == 6:  # genus level
            genus = levels[5]
            if genus and genus != "NA":
                genus_data[col_name] = float(val) if pd.notna(val) else 0.0

    if not genus_data:
        return []

    # 转相对丰度
    total = sum(genus_data.values())
    if total == 0:
        return []

    rel_abundances = {k: v / total for k, v in genus_data.items()}
    sorted_genera = sorted(rel_abundances.items(), key=lambda x: -x[1])

    # 简化属名（取 Bacteria.xxx.yyy.zzz.www.GenusName 中的 GenusName）
    result = []
    for col_name, rel_ab in sorted_genera[:top_n]:
        genus_name = col_name.split(".")[5]
        if rel_ab >= 0.001:  # 过滤 < 0.1% 的
            result.append((genus_name, rel_ab))

    return result


def make_taxon_description(genera_list):
    """将属列表转为自然语言描述"""
    if not genera_list:
        return "（菌群数据不足）"

    parts = [f"{name} ({rel_ab:.2%})" for name, rel_ab in genera_list]
    return "，".join(parts)


def build_qa_samples(df):
    """为每个样本生成 Q&A 对"""
    samples = []
    taxonomy_cols = [c for c in df.columns if c not in [
        "Unnamed: 0", "sample", "run_accession",
        "attribute_ibd", "attribute_ibd_diagnosis_refined",
        "label", "diagnosis_detail"
    ]]

    for idx, (_, row) in enumerate(df.iterrows()):
        genera = extract_top_genera(row)
        if not genera:
            continue

        species_desc = make_taxon_description(genera)
        label = row["label"]
        diagnosis = row["diagnosis_detail"]

        # 构建 labels（区分 binary 和 multi-class）
        binary_label = "Disease" if label != "Healthy" else "Healthy"

        # user message
        user_content = (
            f"你是一位专业的肠道微生物分析师。请分析样本的菌群数据。\n\n"
            f"【主要菌属构成】: {species_desc}\n\n"
            f"请判断该样本的健康状态（Healthy 或 Disease），并简要说明理由。"
        )

        if label == "Healthy":
            assistant_content = (
                f"诊断结果：Healthy。\n\n"
                f"分析理由：该样本菌群构成在正常范围内，各菌属比例均衡，"
                f"未发现明显的菌群失调特征。"
            )
        else:
            assistant_content = (
                f"诊断结果：Disease。\n\n"
                f"分析理由：该样本表现出菌群失调特征，"
                f"其中 {genera[0][0]} 等菌属的丰度变化与肠道炎症相关。"
            )

        samples.append({
            "dataset_type": "agp",
            "sample_id": row["run_accession"],
            "label": binary_label,
            "label_detail": label,
            "diagnosis": diagnosis,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })

        if (idx + 1) % 500 == 0:
            print(f"  处理进度: {idx + 1}/{len(df)}")

    print(f"\n生成 {len(samples)} 个训练样本")
    return samples


def show_data_summary(samples):
    """打印数据概览"""
    labels = Counter(s["label"] for s in samples)
    print(f"\n{'='*60}")
    print(f"数据概览")
    print(f"{'='*60}")
    print(f"总样本数: {len(samples)}")
    print(f"标签分布: {dict(labels)}")

    # 展示几条样本
    print(f"\n样本示例:")
    for i in range(min(3, len(samples))):
        s = samples[i]
        print(f"\n--- 样本 {i+1} ---")
        print(f"  Label: {s['label']} ({s['label_detail']})")
        print(f"  User: {s['messages'][0]['content'][:150]}...")
        print(f"  Assistant: {s['messages'][1]['content'][:100]}...")


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Step 1: 加载并链接数据")
    df = load_and_link_data()

    print("\nStep 2: 过滤样本")
    df = filter_samples(df)

    print("\nStep 3: 生成 Q&A 样本")
    samples = build_qa_samples(df)

    show_data_summary(samples)

    print("\nStep 4: 划分 train / test (80/20)")
    train_data, test_data = train_test_split(
        samples, test_size=0.2, random_state=RANDOM_STATE,
        stratify=[s["label"] for s in samples]
    )
    print(f"  训练集: {len(train_data)}")
    print(f"  测试集: {len(test_data)}")

    print("\nStep 5: 训练集过采样平衡")
    train_labels = Counter(s["label"] for s in train_data)
    max_count = max(train_labels.values())
    grouped = {}
    for s in train_data:
        grouped.setdefault(s["label"], []).append(s)

    balanced_train = []
    for label, samples_list in grouped.items():
        if len(samples_list) == max_count:
            balanced_train.extend(samples_list)
        else:
            times = max_count // len(samples_list)
            remainder = max_count % len(samples_list)
            oversampled = samples_list * times + random.sample(samples_list, remainder)
            balanced_train.extend(oversampled)

    random.shuffle(balanced_train)
    print(f"  平衡后训练集: {len(balanced_train)}")
    print(f"  分布: {dict(Counter(s['label'] for s in balanced_train))}")

    # 保存
    train_path = os.path.join(OUTPUT_DIR, "train_set.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_set.jsonl")

    with open(train_path, "w") as f:
        for s in balanced_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for s in test_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成！")
    print(f"  训练数据: {train_path}")
    print(f"  测试数据: {test_path}")

    # 打印详细的标签分布
    print(f"\n训练集原始标签分布:")
    for k, v in sorted(Counter(s["label_detail"] for s in train_data).items()):
        print(f"  {k}: {v}")
    print(f"\n测试集标签分布:")
    for k, v in sorted(Counter(s["label_detail"] for s in test_data).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
