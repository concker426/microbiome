import torch
import torch.nn as nn
from typing import Dict, Optional


class UnifiedMicrobiomeEncoder(nn.Module):
    """
    统一的微生物组编码器 - Transformer版本
    支持动态输入维度，可处理不同数据集的特征数量
    """
    def __init__(self, num_species=6374, hidden_size=768, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_species = num_species
        self.hidden_size = hidden_size
        
        # 输入投影层 - 将任意维度的OTU向量映射到hidden_size
        self.input_projection = nn.Linear(num_species, hidden_size)
        
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=8, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出归一化和dropout
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, count_vector, return_dict=False):
        """
        Args:
            count_vector: [batch_size, num_species] OTU计数向量
            return_dict: 是否返回字典格式
        Returns:
            embedding: [batch_size, hidden_size] 微生物群落embedding
        """
        # 输入投影: [batch, num_species] -> [batch, hidden_size]
        x = self.input_projection(count_vector)
        
        # Transformer需要3D输入: [batch, seq_len, hidden]
        # 这里seq_len=1，因为每个样本是一个整体向量
        x = x.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch, 1, hidden_size]
        
        # 去掉seq_len维度
        x = x.squeeze(1)  # [batch, hidden_size]
        
        # 归一化和dropout
        x = self.output_norm(x)
        x = self.dropout(x)
        
        if return_dict:
            return {
                "last_hidden_state": x,
                "pooler_output": x,
            }
        return x
    
    @classmethod
    def create_for_dataset(cls, dataset_type="study", hidden_size=768, num_layers=2, dropout=0.1):
        """
        根据数据集类型创建对应维度的编码器
        
        Args:
            dataset_type: "study" (6374维) 或 "ibd" (300维)
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            dropout: dropout比率
        """
        if dataset_type == "study":
            num_species = 6374  # Study数据集有6374个物种特征
        elif dataset_type == "ibd":
            num_species = 300   # IBD数据集有300个OTU特征
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        return cls(
            num_species=num_species,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )


class ProjectionLayer(nn.Module):
    """
    投影层 - 将微生物embedding映射到LLM的隐藏空间
    """
    def __init__(self, input_dim=768, output_dim=3584):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


def load_encoder_with_weights(encoder, weights_path, device="cuda:0"):
    """
    加载预训练的encoder权重
    
    Args:
        encoder: MicrobiomeEncoder实例
        weights_path: 权重文件路径
        device: 设备
    """
    checkpoint = torch.load(weights_path, map_location=device)
    if 'micro_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['micro_encoder'])
    else:
        encoder.load_state_dict(checkpoint)
    print(f"✅ 已加载encoder权重: {weights_path}")
    return encoder


if __name__ == "__main__":
    # 测试不同维度的编码器
    print("测试UnifiedMicrobiomeEncoder...")
    
    # 测试Study数据集 (6374维)
    encoder_study = UnifiedMicrobiomeEncoder.create_for_dataset("study")
    test_input_study = torch.randn(2, 6374)
    output_study = encoder_study(test_input_study)
    print(f"Study数据集: 输入{test_input_study.shape} -> 输出{output_study.shape}")
    
    # 测试IBD数据集 (300维)
    encoder_ibd = UnifiedMicrobiomeEncoder.create_for_dataset("ibd")
    test_input_ibd = torch.randn(2, 300)
    output_ibd = encoder_ibd(test_input_ibd)
    print(f"IBD数据集: 输入{test_input_ibd.shape} -> 输出{output_ibd.shape}")
    
    print("✅ 所有测试通过！")
