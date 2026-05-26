import pandas as pd
import json
import os
import numpy as np

# 配置路径
DATA_DIR = "/hd/liujx/microbiome_llm_project/data"
COUNTS_FILE = os.path.join(DATA_DIR, "study_16496_counts.tsv")
OUTPUT_FILE = os.path.join(DATA_DIR, "microbiome_qa_train.jsonl")

def generate_qa_dataset():
    print(f"正在读取数据: {COUNTS_FILE} ...")
    
    # 关键修改：skiprows=2 跳过前两行注释，header=0 让第三行做表头
    df = pd.read_csv(COUNTS_FILE, sep='\t', index_col=0, skiprows=2, low_memory=False)
    
    # 检查行列结构：如果行数是物种数(6000+)，列数是样本数(400+)，则需要转置
    print(f"原始数据形状: {df.shape}")
    if df.shape[0] > df.shape[1]:
        print("检测到行多于列（物种 x 样本），正在转置为（样本 x 物种）...")
        df = df.T
    
    samples = []
    total = len(df)
    print(f"开始处理 {total} 个样本...")
    
    for idx, (sample_id, row) in enumerate(df.iterrows()):
        # 确保数据是数值型
        row = pd.to_numeric(row, errors='coerce').fillna(0)
        
        # 1. 提取 Top 5 优势菌种
        top5 = row.nlargest(5)
        species_desc = ", ".join([f"{name} ({val:.2%})" for name, val in top5.items()])
        
        # 2. 构造标签 (简单逻辑：包含 BL 为 Healthy，否则为 IBD)
        label = "Healthy" if "BL" in str(sample_id) else "IBD"
        
        # 3. 构造 User 提问
        user_content = (
            f"你是一位专业的肠道微生物分析师。请分析样本 {sample_id} 的菌群数据。\n\n"
            f"【主要菌群构成】: {species_desc}\n"
            f"请判断该样本的健康状态（Healthy 或 IBD），并简要说明理由。"
        )
        
        # 4. 构造 Assistant 回答
        if label == "IBD":
            assistant_content = (
                f"诊断结果：IBD（炎症性肠病）。\n\n"
                f"分析理由：该样本显示出典型的菌群失调特征。"
                f"优势菌种为 {top5.index[0]}，其异常丰度通常与肠道炎症相关。"
            )
        else:
            assistant_content = (
                f"诊断结果：Healthy（健康）。\n\n"
                f"分析理由：该样本菌群多样性良好，各菌门比例处于正常范围，"
                f"未发现明显的致病菌富集现象。"
            )
            
        # 5. 组装成 JSONL
        conversation = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        samples.append(conversation)
        
        if (idx + 1) % 50 == 0:
            print(f"进度: {idx + 1}/{total}")

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 语料生成完毕！文件保存在: {OUTPUT_FILE}")
    print(f"   总样本数: {len(samples)}")

if __name__ == "__main__":
    generate_qa_dataset()
