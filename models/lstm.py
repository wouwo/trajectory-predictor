import torch
import torch.nn as nn
from config import Config

class TrajectoryLSTM(nn.Module):
    def __init__(self):
        """
        基于 LSTM 的轨迹预测模型
        """
        super(TrajectoryLSTM, self).__init__()
        
        hidden_size = Config.LSTM_HIDDEN_SIZE
        num_layers = Config.LSTM_LAYERS
        
        # 1. 嵌入层: Input -> Hidden
        self.embedding = nn.Linear(Config.INPUT_SIZE, hidden_size)
        
        # 2. LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1
        )
        
        # 3. 解码器: Hidden -> Output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, Config.PRED_LEN * Config.OUTPUT_SIZE)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Obs_Len, 4]
        Returns:
            out: [Batch, Pred_Len, 4]
        """
        batch_size = x.size(0)
        
        # [Batch, Obs_Len, 2] -> [Batch, Obs_Len, Hidden]
        x = self.embedding(x)
        
        # LSTM 前向传播
        # out: [Batch, Seq, Hidden], (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        
        # 取最后一层的最后一个时间步状态: [Batch, Hidden]
        # h_n 的形状是 [num_layers, batch, hidden_size]
        last_hidden = h_n[-1, :, :]
        
        # 解码预测
        out = self.decoder(last_hidden)
        
        # 重塑为轨迹形状: [Batch, Pred_Len, 2]
        out = out.view(batch_size, Config.PRED_LEN, Config.OUTPUT_SIZE)
        
        return out