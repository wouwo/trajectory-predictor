import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000): 
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TrajectoryTransformer(nn.Module):
    def __init__(self):
        """
        基于 Transformer 的轨迹预测模型
        """
        super(TrajectoryTransformer, self).__init__()
        
        # 获取 Transformer 专属配置
        d_model = Config.TRANS_D_MODEL
        nhead = Config.TRANS_NHEAD
        num_layers = Config.TRANS_LAYERS
        dim_feedforward = Config.TRANS_FF_DIM
        dropout = Config.TRANS_DROPOUT
        
        # 1. 嵌入与位置编码
        self.input_fc = nn.Linear(Config.INPUT_SIZE, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 3. 解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, Config.PRED_LEN * Config.OUTPUT_SIZE)
        )

    def forward(self, src):
        """
        Args:
            src: [Batch, Obs_Len, 2]
        """
        batch_size = src.size(0)
        
        # 1. 维度转换 [Batch, Seq, Dim] -> [Seq, Batch, Dim] (Transformer 默认输入)
        src = src.permute(1, 0, 2)
        
        # 2. 嵌入
        x = self.input_fc(src)
        x = self.pos_encoder(x)
        
        # 3. 编码
        memory = self.transformer_encoder(x)
        
        # 4. 聚合特征: 取最后一个时间步 [Batch, Dim]
        last_hidden = memory[-1, :, :]
        
        # 5. 解码
        out = self.decoder(last_hidden)
        
        # 6. 重塑 [Batch, Pred_Len, 2]
        out = out.view(batch_size, Config.PRED_LEN, Config.OUTPUT_SIZE)
        
        return out