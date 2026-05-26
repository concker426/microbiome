"""
数据合并脚本 - 将Study和IBD两个数据集合并用于训练
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List


def load_study_dataset(counts_file="/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"):
    """加载Study数据集 (421维特征)"""
    print("📂 加载Study数据集...")
    df = pd.read_csv(counts_file, sep='\t', index_col=0, skiprows=2, low_memory=False)
    if df.shape[0] > df.shape[1]:
        df = df.T
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Study数据集没有明确的标签，根据样本ID推断
    labels = {}
    for sample_id in df.index:
        # BL开头的可能是baseline/healthy，其他可能是疾病
        label = "Healthy" if "BL" in str(sample_id).upper() else "IBD"
        labels[str(sample_id)] = label
    
    print(f"✅ Study数据集: {df.shape[0]} 样本 × {df.shape[1]} 特征")
    return df, labels, "study"


def load_ibd_dataset(
    counts_file="/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv",
    metadata_file="/hd/liujx/microbiome_llm_project/data/ibd_metadata.txt"
):
    """加载IBD数据集 (300维特征)"""
    print("📂 加载IBD数据集...")
    df = pd.read_csv(counts_file, sep='\t', index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 从元数据文件加载细粒度标签
    labels = {}
    if os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file, sep='\t', index_col=0)
        for sample_id, row in meta_df.iterrows():
            disease = str(row['Disease']).strip()
            labels[str(sample_id)] = disease
    
    print(f"✅ IBD数据集: {df.shape[0]} 样本 × {df.shape[1]} 特征")
    print(f"   标签分布: {pd.Series(labels).value_counts().to_dict()}")
    return df, labels, "ibd"


def normalize_and_merge_datasets(study_df, ibd_df):
    """
    标准化并合并两个数据集
    由于特征维度不同，我们分别保存，但在训练时统一处理
    """
    print("\n🔄 标准化数据集...")
    
    # 对每个数据集进行log1p转换
    study_normalized = np.log1p(study_df)
    ibd_normalized = np.log1p(ibd_df)
    
    print(f"Study数据集范围: [{study_normalized.min().min():.2f}, {study_normalized.max().max():.2f}]")
    print(f"IBD数据集范围: [{ibd_normalized.min().min():.2f}, {ibd_normalized.max().max():.2f}]")
    
    return study_normalized, ibd_normalized


def generate_merged_training_data(
    output_jsonl="/hd/liujx/microbiome_llm_project/data/merged_training_data.jsonl",
    samples_per_dataset=None
):
    """
    生成合并的训练数据JSONL文件
    
    Args:
        output_jsonl: 输出文件路径
        samples_per_dataset: 每个数据集使用的样本数（None表示全部）
    """
    print("="*60)
    print("开始合并数据集...")
    print("="*60)
    
    # 加载两个数据集
    study_df, study_labels, study_type = load_study_dataset()
    ibd_df, ibd_labels, ibd_type = load_ibd_dataset()
    
    # 标准化
    study_norm, ibd_norm = normalize_and_merge_datasets(study_df, ibd_df)
    
    # 限制样本数（如果指定）
    if samples_per_dataset:
        study_samples = study_norm.iloc[:samples_per_dataset]
        ibd_samples = ibd_norm.iloc[:samples_per_dataset]
        print(f"\n⚠️  限制每个数据集使用 {samples_per_dataset} 个样本")
    else:
        study_samples = study_norm
        ibd_samples = ibd_norm
    
    # 生成训练样本
    all_samples = []
    
    print("\n📝 生成Study数据集训练样本...")
    for idx, (sample_id, row) in enumerate(study_samples.iterrows()):
        label = study_labels.get(str(sample_id), "IBD")
        
        # 获取top物种
        top_n = row.nlargest(5)
        top_species = ", ".join([f"{name[:50]} ({val:.2%})" for name, val in top_n.items()])
        
        # 生成QA样本
        qa_sample = {
            "task_type": "qa",
            "dataset_type": "study",
            "sample_id": str(sample_id),
            "label": label,
            "messages": [
                {
                    "role": "user",
                    "content": f"你是一位专业的肠道微生物分析师。请分析样本 {str(sample_id)[:20]} 的菌群数据。\n\n【主要菌群构成】: {top_species}\n\n请判断该样本的健康状态（Healthy 或 IBD），并简要说明理由。"
                },
                {
                    "role": "assistant",
                    "content": f"诊断结果：{label}。\n\n分析理由：基于微生物组数据分析，该样本{'显示正常的菌群多样性，各菌门比例处于健康范围' if label == 'Healthy' else '显示出菌群失调特征，与炎症性肠病相关'}。"
                }
            ]
        }
        all_samples.append(qa_sample)
        
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx + 1}/{len(study_samples)}")
    
    print("\n📝 生成IBD数据集训练样本...")
    for idx, (sample_id, row) in enumerate(ibd_samples.iterrows()):
        label = ibd_labels.get(str(sample_id), "IBD")
        
        # 获取top物种
        top_n = row.nlargest(5)
        top_species = ", ".join([f"OTU_{name.split('_')[1] if '_' in name else name} ({val:.2%})" for name, val in top_n.items()])
        
        # 生成多分类样本（支持CD/UC/IBD/Healthy）
        qa_sample = {
            "task_type": "qa",
            "dataset_type": "ibd",
            "sample_id": str(sample_id),
            "label": label,
            "messages": [
                {
                    "role": "user",
                    "content": f"你是一位专业的肠道微生物分析师。请分析样本 {sample_id} 的菌群数据。\n\n【主要菌群构成】: {top_species}\n\n请判断该样本的健康状态（Healthy、IBD、CD或UC），并简要说明理由。"
                },
                {
                    "role": "assistant",
                    "content": f"诊断结果：{label}。\n\n分析理由：基于微生物组数据分析，该样本{'显示正常的菌群多样性' if label == 'Healthy' else '显示出菌群失调特征'}。"
                }
            ]
        }
        all_samples.append(qa_sample)
        
        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{len(ibd_samples)}")
    
    # 保存为JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 合并训练数据生成完毕！")
    print(f"   输出文件: {output_jsonl}")
    print(f"   总样本数: {len(all_samples)}")
    
    # 统计信息
    dataset_counts = {}
    label_counts = {}
    for sample in all_samples:
        ds = sample['dataset_type']
        lbl = sample['label']
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    
    print(f"   数据集分布: {dataset_counts}")
    print(f"   标签分布: {label_counts}")
    
    # 保存标签映射
    label_file = output_jsonl.replace('.jsonl', '_labels.json')
    all_labels = {**study_labels, **ibd_labels}
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    print(f"   标签文件: {label_file}")
    
    return output_jsonl


if __name__ == "__main__":
    generate_merged_training_data(samples_per_dataset=None)
