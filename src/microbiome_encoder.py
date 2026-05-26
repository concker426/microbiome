

import torch
import torch.nn as nn

class MicrobiomeEncoder(nn.Module):
    """
    微生物计数表编码器
    将物种计数向量映射到 embedding 空间
    """
    def __init__(self, num_species, hidden_size=768, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_species = num_species
        self.hidden_size = hidden_size
        
        # 输入投影层
        self.input_projection = nn.Linear(num_species, hidden_size)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, count_vector):
        """
        Args:
            count_vector: [batch_size, num_species] 计数向量
        Returns:
            embedding: [batch_size, hidden_size] 微生物群落 embedding
        """
        x = self.input_projection(count_vector)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        return self.output_norm(x.squeeze(1))
