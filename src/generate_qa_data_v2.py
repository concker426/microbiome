import pandas as pd
import json
import os
import glob
import numpy as np

# 配置路径
DATA_DIR = "/hd/liujx/microbiome_llm_project/data"
COUNTS_FILE = os.path.join(DATA_DIR, "study_16496_counts.tsv")
MAPPING_DIR = os.path.join(DATA_DIR, "study_16496_040726-180652/mapping_files")
OUTPUT_FILE = os.path.join(DATA_DIR, "microbiome_qa_enhanced.jsonl")

def load_metadata(mapping_dir):
    """合并所有 mapping 文件并去重"""
    all_meta = []
    for f in glob.glob(os.path.join(mapping_dir, "*_mapping_file.txt")):
        try:
            df = pd.read_csv(f, sep='\t', low_memory=False)
            all_meta.append(df)
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    if all_meta:
        merged = pd.concat(all_meta, ignore_index=True)
        # 关键修复：如果 sample_name 重复，保留第一个出现的
        if 'sample_name' in merged.columns:
            merged = merged.drop_duplicates(subset='sample_name', keep='first')
        return merged
    return pd.DataFrame()

def generate_enhanced_qa():
    print("1. 正在加载 OTU 计数表...")
    df_counts = pd.read_csv(COUNTS_FILE, sep='\t', index_col=0, skiprows=2, low_memory=False)
    if df_counts.shape[0] > df_counts.shape[1]:
        df_counts = df_counts.T
    
    print("2. 正在加载元数据 (Mapping files)...")
    df_meta = load_metadata(MAPPING_DIR)
    
    # 建立样本 ID 到元数据的映射
    meta_dict = {}
    if not df_meta.empty and 'sample_name' in df_meta.columns:
        df_meta.set_index('sample_name', inplace=True)
        meta_dict = df_meta.to_dict(orient='index')

    samples = []
    total = len(df_counts)
    print(f"3. 开始处理 {total} 个样本...")
    
    for idx, (sample_id, row) in enumerate(df_counts.iterrows()):
        row = pd.to_numeric(row, errors='coerce').fillna(0)
        top5 = row.nlargest(5)
        species_desc = ", ".join([f"{name} ({val:.2%})" for name, val in top5.items()])
        
        # 获取元数据
        meta_info = meta_dict.get(str(sample_id), {})
        exp_desc = meta_info.get('experiment_design_description', '未知实验背景')
        if len(exp_desc) > 200: exp_desc = exp_desc[:200] + "..."
        
        label = "Healthy" if "BL" in str(sample_id) else "IBD"
        
        # 构造更丰富的 Prompt
        user_content = (
            f"你是一位微生物组专家。请分析样本 {sample_id}。\n\n"
            f"【实验背景】: {exp_desc}\n"
            f"【主要菌群】: {species_desc}\n\n"
            f"请判断该样本的健康状态（Healthy 或 IBD），并说明理由。"
        )
        
        if label == "IBD":
            assistant_content = f"诊断结果：IBD。理由：基于实验背景及菌群失调特征（如 {top5.index[0]} 异常升高），判定为炎症性肠病。"
        else:
            assistant_content = f"诊断结果：Healthy。理由：菌群构成均衡，多样性良好，符合健康基线特征。"
            
        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })
        
        if (idx + 1) % 50 == 0:
            print(f"进度: {idx + 1}/{total}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 增强版语料已保存: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_enhanced_qa()
