import os
import re
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional

# ================= 全局配置 (模仿 ProCyon constants) =================
PROJECT_ROOT = "/hd/liujx/microbiome_llm_project"
DEFAULT_COUNTS_FILE = os.path.join(PROJECT_ROOT, "data/study_16496_counts.tsv")
IBD_COUNTS_FILE = os.path.join(PROJECT_ROOT, "data/ibd_counts.tsv")
NUM_SPECIES = 6374

class MicrobiomeDataLoader:
    """
    模仿 ProCyon 的数据加载器。
    负责：读取 TSV -> 转置 -> 索引对齐 -> 提供向量查询。
    """
    def __init__(self, counts_file: str = None, skiprows: int = 2):
        self.counts_file = counts_file or DEFAULT_COUNTS_FILE
        print(f"🔄 [DataLoader] 正在从 {self.counts_file} 加载数据...")
        # 模仿 ProCyon 的 read_pickle/read_csv 预处理
        df = pd.read_csv(self.counts_file, sep='\t', index_col=0, skiprows=skiprows, low_memory=False, dtype=str)
        
        # 智能转置逻辑：根据文件名判断是否需要转置
        if "ibd_counts" in self.counts_file:
            # ibd_counts.tsv 已经是样本×特征格式，无需转置
            pass
        else:
            # study_16496_counts.tsv 需要转置为样本×特征格式
            if df.shape[1] != NUM_SPECIES:
                df = df.T
        
        # 索引清洗
        cleaned_index = []
        for idx in df.index.astype(str):
            idx = idx.strip()
            if re.fullmatch(r"\d+\.0", idx):
                idx = str(int(float(idx)))
            cleaned_index.append(idx)
        df.index = cleaned_index

        self.otu_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        print(f"✅ [DataLoader] 就绪: {len(self.otu_df)} samples x {self.otu_df.shape[1]} features")

    def get_sample_vector(self, sample_id: str) -> Optional[torch.Tensor]:
        """模仿 ProCyon 的 uniprot_id_to_index + 序列获取"""
        sid = str(sample_id).strip()

        # 直接匹配
        if sid in self.otu_df.index:
            vec = self.otu_df.loc[sid].values.astype(np.float32)
            return torch.tensor(vec).unsqueeze(0) # [1, 6374]

        # 尝试数值转化：4016 -> 4016.0
        try:
            sid_float = float(sid)
            if sid_float.is_integer():
                sid_alt = str(int(sid_float))
                if sid_alt in self.otu_df.index:
                    vec = self.otu_df.loc[sid_alt].values.astype(np.float32)
                    return torch.tensor(vec).unsqueeze(0)
            sid_alt = str(sid_float)
            if sid_alt in self.otu_df.index:
                vec = self.otu_df.loc[sid_alt].values.astype(np.float32)
                return torch.tensor(vec).unsqueeze(0)
        except ValueError:
            pass

        return None

class MicrobiomeInputConstructor:
    """
    模仿 ProCyon 的 create_qa_input_simple。
    负责：构造包含 text, seq, instructions 的标准输入字典。
    """
    def __init__(self, tokenizer, device="cuda:0"):
        self.tokenizer = tokenizer
        self.device = device

    def create_diagnosis_input(self, sample_id: str, vector: torch.Tensor) -> Dict:
        # 1. 构造指令 (这里可以后续替换为从 JSON 读取)
        prompt = (
            f"你是一位专业的肠道微生物分析师。请分析样本 {sample_id} 的菌群数据。\n\n"
            f"【主要菌群构成】: 请参考提供的数值向量。\n\n"
            f"请判断该样本的健康状态（Healthy 或 IBD），并简要说明理由。"
        )
        
        # 2. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 3. 返回标准字典 (结构与 ProCyon 的 model_input 保持一致)
        return {
            "data": {
                "seq": vector.to(self.device),   # 对应 ProCyon 的 seq (蛋白质/微生物)
                "text": [prompt],                # 对应 ProCyon 的 text
            },
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "sample_id": sample_id
        }
