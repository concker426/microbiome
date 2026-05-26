import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class MicrobiomeEncoderV2(nn.Module):
    def __init__(self, num_species=6374, hidden_size=768, num_layers=2, dropout=0.1, max_length=1024):
        super().__init__()
        self.num_species = num_species
        self.hidden_size = hidden_size
        
        # 输入投影
        self.input_projection = nn.Linear(num_species, hidden_size)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=8, 
            dim_feedforward=hidden_size*4, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, count_vector, return_dict=False):
        """
        Args:
            count_vector: [batch_size, num_species]
        Returns:
            embedding: [batch_size, hidden_size]
        """
        # 输入投影: [batch, num_species] -> [batch, hidden_size]
        x = self.input_projection(count_vector)
        
        # Transformer 需要 3D 输入: [batch, seq_len, hidden]
        # 这里 seq_len=1，因为每个样本是一个整体向量
        x = x.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # [batch, 1, hidden_size]
        
        # 去掉 seq_len 维度
        x = x.squeeze(1)  # [batch, hidden_size]
        
        # 归一化和 dropout
        x = self.output_norm(x)
        x = self.dropout(x)
        
        if return_dict:
            return {
                "last_hidden_state": x,
                "pooler_output": x,
            }
        return x


class ProjectionLayer(nn.Module):
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
