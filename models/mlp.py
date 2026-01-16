import torch
import torch.nn as nn
from config import Config

class TrajectoryMLP(nn.Module):
    def __init__(self):
        super(TrajectoryMLP, self).__init__()
        
        # 输入维度: 观察长度 * 特征数 (例如 8 * 4 = 16)
        self.input_dim = Config.OBS_LEN * Config.INPUT_SIZE
        
        # 输出维度: 预测长度 * 特征数 (例如 12 * 2 = 24)
        self.output_dim = Config.PRED_LEN * Config.OUTPUT_SIZE
        
        hidden_dim = Config.MLP_HIDDEN_SIZE
        
        # 定义多层感知机结构
        # 使用 BatchNorm 和 ReLU 增加非线性表达能力
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, obs_len, input_size] (例如 [64, 8, 4])
        Returns:
            out: shape [batch_size, pred_len, output_size] (例如 [64, 12, 2])
        """
        batch_size = x.size(0)
        
        # 1. Flatten: 将时间维度展平 [B, 8, 2] -> [B, 16]
        x = x.view(batch_size, -1)
        
        # 2. Forward pass
        out = self.net(x)
        
        # 3. Reshape: 恢复时间维度 [B, 24] -> [B, 12, 2]
        out = out.view(batch_size, Config.PRED_LEN, Config.OUTPUT_SIZE)
        
        return out