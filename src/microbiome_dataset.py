import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

class MicrobiomeDataset(Dataset):
    """
    微生物计数表数据集
    """
    def __init__(self, count_table_path, config=None):
        print(f"📂 加载数据：{count_table_path}")
        if not os.path.exists(count_table_path):
            raise FileNotFoundError(f"文件不存在：{count_table_path}")
        
        self.count_table = pd.read_csv(count_table_path, index_col=0).fillna(0)
        print(f"✅ 加载成功：{len(self.count_table)} 样本，{self.count_table.shape[1]} 物种")
        
        self.count_table = np.log1p(self.count_table)
        self.config = config
        
    def __len__(self):
        return len(self.count_table)
    
    def __getitem__(self, idx):
        count_vector = self.count_table.iloc[idx].values.astype(np.float32)
        return {
            'count_vector': torch.from_numpy(count_vector),
            'sample_id': str(self.count_table.index[idx]),
        }