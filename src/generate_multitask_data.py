import pandas as pd
import json
import os
import numpy as np
from typing import List, Dict

RETRIEVAL_TEMPLATES = [
    {"instruction": "根据以下微生物组特征，检索相关的疾病描述。", "input_format": "样本微生物组成: {top_species}", "output_format": "{disease_description}"},
    {"instruction": "这个微生物组模式与哪种肠道疾病最相关？", "input_format": "优势菌种: {top_species}", "output_format": "{disease_name}"},
    {"instruction": "请识别该微生物组样本的临床表型。", "input_format": "菌群构成: {top_species}", "output_format": "{phenotype}"}
]

QA_TEMPLATES = [
    {"instruction": "判断该样本是否患有炎症性肠病(IBD)。回答 yes 或 no。", "input_format": "样本ID: {sample_id}\n主要菌群: {top_species}", "output_format": "{yes_or_no}"},
    {"instruction": "这个微生物组样本健康吗？只回答 yes 或 no。", "input_format": "菌群构成: {top_species}", "output_format": "{yes_or_no}"},
    {"instruction": "该样本是否显示菌群失调特征？回答 yes 或 no。", "input_format": "样本 {sample_id} 的菌群数据: {top_species}", "output_format": "{yes_or_no}"}
]

GENERATION_TEMPLATES = [
    {"instruction": "你是一位专业的肠道微生物分析师。请分析以下样本的菌群数据。", "input_format": "样本ID: {sample_id}\nTop 5优势菌种: {top_species}\n实验背景: {metadata}", "output_format": "{detailed_analysis}"},
    {"instruction": "请详细解释这个微生物组样本的临床意义。", "input_format": "样本信息: {top_species}", "output_format": "{clinical_interpretation}"},
    {"instruction": "基于微生物组数据，生成一份诊断报告。", "input_format": "患者样本 {sample_id}\n菌群构成: {top_species}", "output_format": "{diagnosis_report}"}
]

def load_and_prepare_data(counts_file):
    print(f"正在读取数据: {counts_file} ...")
    df = pd.read_csv(counts_file, sep='\t', index_col=0, skiprows=2, low_memory=False)
    if df.shape[0] > df.shape[1]:
        print("检测到行多于列，正在转置...")
        df = df.T
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"数据形状: {df.shape}")
    return df

def get_top_species(row, n=5):
    top_n = row.nlargest(n)
    return ", ".join([f"{name} ({val:.2%})" for name, val in top_n.items()])

def determine_label(sample_id):
    return "Healthy" if "BL" in str(sample_id) else "IBD"

def generate_detailed_report(sample_id, top_species, label):
    if label == "IBD":
        first_species = top_species.split(',')[0]
        return f"诊断结果：IBD（炎症性肠病）。\n\n分析理由：该样本显示出典型的菌群失调特征。优势菌种为 {first_species}，其异常丰度通常与肠道炎症相关。建议进一步进行临床检查以确认诊断。"
    else:
        return f"诊断结果：Healthy（健康）。\n\n分析理由：该样本菌群多样性良好，各菌门比例处于正常范围，未发现明显的致病菌富集现象。整体微生物组结构稳定。"

def create_retrieval_sample(sample_id, row, label, template):
    top_species = get_top_species(row, n=5)
    user_content = template['instruction'] + "\n\n" + template['input_format'].format(top_species=top_species)
    assistant_content = template['output_format'].format(disease_description=f"{label} disease associated microbiome pattern", disease_name=label, phenotype=f"Patient with {label}")
    return {"task_type": "retrieval", "sample_id": str(sample_id), "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}], "positive_text": f"{label} disease microbiome", "negative_text": "unrelated healthy microbiome"}

def create_qa_sample(sample_id, row, label, template):
    top_species = get_top_species(row, n=5)
    yes_or_no = "yes" if label == "IBD" else "no"
    user_content = template['instruction'] + "\n\n" + template['input_format'].format(sample_id=sample_id, top_species=top_species)
    return {"task_type": "qa", "sample_id": str(sample_id), "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": yes_or_no}]}

def create_generation_sample(sample_id, row, label, template):
    top_species = get_top_species(row, n=5)
    user_content = template['instruction'] + "\n\n" + template['input_format'].format(sample_id=sample_id, top_species=top_species, metadata="Qiita Study 16496")
    assistant_content = generate_detailed_report(sample_id, top_species, label)
    return {"task_type": "generation", "sample_id": str(sample_id), "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]}

def generate_multitask_dataset(counts_file, output_file, samples_per_task=1):
    df = load_and_prepare_data(counts_file)
    all_samples = []
    total = len(df)
    print(f"开始生成多任务数据集 (共 {total} 个样本)...")
    for idx, (sample_id, row) in enumerate(df.iterrows()):
        label = determine_label(sample_id)
        for _ in range(samples_per_task):
            retrieval_template = np.random.choice(RETRIEVAL_TEMPLATES)
            qa_template = np.random.choice(QA_TEMPLATES)
            generation_template = np.random.choice(GENERATION_TEMPLATES)
            retrieval_sample = create_retrieval_sample(sample_id, row, label, retrieval_template)
            qa_sample = create_qa_sample(sample_id, row, label, qa_template)
            generation_sample = create_generation_sample(sample_id, row, label, generation_template)
            all_samples.extend([retrieval_sample, qa_sample, generation_sample])
        if (idx + 1) % 50 == 0:
            print(f"进度: {idx + 1}/{total}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n✅ 多任务语料生成完毕！")
    print(f"   文件: {output_file}")
    print(f"   总样本数: {len(all_samples)}")
    task_counts = {}
    for sample in all_samples:
        task_type = sample['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"   任务分布: {task_counts}")

if __name__ == "__main__":
    DATA_DIR = "/hd/liujx/microbiome_llm_project/data"
    COUNTS_FILE = os.path.join(DATA_DIR, "study_16496_counts.tsv")
    OUTPUT_FILE = os.path.join(DATA_DIR, "microbiome_multitask.jsonl")
    generate_multitask_dataset(counts_file=COUNTS_FILE, output_file=OUTPUT_FILE, samples_per_task=1)