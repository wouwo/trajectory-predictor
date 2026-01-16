import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import argparse  # 引入参数解析库

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # 路径配置
    DATA_PATH = 'ewap_dataset/seq_eth/obsmat.txt' 
    MODEL_PATH = 'trajectory_model.pth' # 模型保存路径

    # 轨迹参数
    OBS_LEN = 8    # 观察过去 8 帧 (3.2秒)
    PRED_LEN = 12  # 预测未来 12 帧 (4.8秒)
    
    # 模型参数
    INPUT_SIZE = 2 
    MODEL_DIM = 64 
    NUM_HEADS = 4  
    NUM_LAYERS = 2 
    DROPOUT = 0.1
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 1000
    LR = 0.001
    
    # 设备配置
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

# 固定随机种子，保证训练集和测试集划分一致
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print(f"正在使用设备: {Config.DEVICE}")

# ==========================================
# 2. 数据处理 (Data Loading)
# ==========================================
class ETHDataset(Dataset):
    def __init__(self, file_path, obs_len=8, pred_len=12):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        try:
            # 适配 obsmat.txt
            df = pd.read_csv(file_path, sep='\s+', header=None)
            df = df.iloc[:, [0, 1, 2, 4]] 
            df.columns = ["frame_id", "ped_id", "x", "y"]
        except Exception as e:
            print(f"数据读取失败: {e}")
            raise

        self.data = []
        df['frame_id'] = df['frame_id'].astype(int)
        df['ped_id'] = df['ped_id'].astype(int)

        for ped_id, group in df.groupby("ped_id"):
            group = group.sort_values("frame_id")
            path = group[["x", "y"]].values
            
            if len(path) < self.seq_len:
                continue

            for i in range(len(path) - self.seq_len + 1):
                full_seq = path[i : i + self.seq_len]
                rel_seq = np.zeros_like(full_seq)
                rel_seq[1:] = full_seq[1:] - full_seq[:-1]
                
                self.data.append({
                    "abs_seq": torch.FloatTensor(full_seq),
                    "rel_seq": torch.FloatTensor(rel_seq)
                })
        # print(f"数据集加载完成，共提取 {len(self.data)} 条轨迹片段。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["rel_seq"][:self.obs_len]
        tgt = item["rel_seq"][self.obs_len:]
        start_pos = item["abs_seq"][self.obs_len-1]
        true_path = item["abs_seq"]
        return src, tgt, start_pos, true_path

# ==========================================
# 3. 模型定义 (Transformer Model)
# ==========================================
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
    def __init__(self, config):
        super(TrajectoryTransformer, self).__init__()
        self.config = config
        
        self.input_fc = nn.Linear(config.INPUT_SIZE, config.MODEL_DIM)
        self.pos_encoder = PositionalEncoding(config.MODEL_DIM)
        
        # 注意: 这里的 batch_first 设为 False 以匹配 permute 后的形状 [Seq, Batch, Dim]
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.MODEL_DIM, 
            nhead=config.NUM_HEADS, 
            dim_feedforward=128,
            dropout=config.DROPOUT,
            batch_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.NUM_LAYERS)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.MODEL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, config.PRED_LEN * config.INPUT_SIZE)
        )

    def forward(self, src):
        # Input: [Batch, Seq, Dim] -> Permute -> [Seq, Batch, Dim]
        src = src.permute(1, 0, 2) 
        
        x = self.input_fc(src)       
        x = self.pos_encoder(x)      
        memory = self.transformer_encoder(x) 
        
        last_hidden = memory[-1, :, :] # [Batch, Dim]
        
        prediction_vector = self.decoder(last_hidden) 
        output = prediction_vector.view(-1, self.config.PRED_LEN, self.config.INPUT_SIZE)
        
        return output

def get_dataloaders():
    """辅助函数：获取数据加载器"""
    try:
        dataset = ETHDataset(Config.DATA_PATH, Config.OBS_LEN, Config.PRED_LEN)
    except Exception as e:
        print(f"数据加载出错: {e}")
        return None, None

    if len(dataset) == 0:
        print("错误：数据集为空。")
        return None, None

    # 固定划分比例
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, test_loader

def train():
    set_seed(42) # 训练时固定种子
    print(f"\n[模式] 训练模式 (Training Mode)")
    
    train_loader, test_loader = get_dataloaders()
    if not train_loader: return

    model = TrajectoryTransformer(Config).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()
    
    print("开始训练...")
    loss_history = []
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt, _, _) in enumerate(train_loader):
            src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {avg_loss:.6f}")

    # 保存模型权重
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"\n✅ 模型已保存至: {Config.MODEL_PATH}")

    # 绘制 Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.show()

    # 训练结束后顺便看一眼效果
    visualize_results(model, test_loader)

def test():
    set_seed(42) # 测试时使用相同的种子，确保测试集与训练时的划分一致
    print(f"\n[模式] 测试模式 (Test Mode)")
    
    # 检查模型文件是否存在
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 {Config.MODEL_PATH}")
        print("请先运行训练模式：python script.py --mode train")
        return

    # 获取数据 (我们需要 test_loader)
    _, test_loader = get_dataloaders()
    if not test_loader: return

    # 初始化模型结构
    model = TrajectoryTransformer(Config).to(Config.DEVICE)
    
    # 加载权重
    print(f"正在加载模型: {Config.MODEL_PATH} ...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    print("✅ 模型加载成功！")
    
    # 运行可视化
    visualize_results(model, test_loader)

def visualize_results(model, loader):
    model.eval()
    print("\n正在可视化预测结果...")
    
    with torch.no_grad():
        count = 0
        plt.figure(figsize=(15, 5))
        
        for src, tgt, start_pos, true_abs_path in loader:
            if count >= 3: break
            
            src = src.to(Config.DEVICE)
            pred_rel = model(src) 
            
            pred_rel = pred_rel.cpu().numpy()[0]
            start_pos = start_pos.numpy()[0]
            true_abs_path = true_abs_path.numpy()[0]
            
            # --- 坐标还原 ---
            pred_abs = [start_pos] 
            curr = start_pos
            for i in range(len(pred_rel)):
                curr = curr + pred_rel[i] 
                pred_abs.append(curr)
            pred_abs = np.array(pred_abs)
            
            # --- 绘图 ---
            plt.subplot(1, 3, count+1)
            obs_len = Config.OBS_LEN
            
            plt.plot(true_abs_path[:obs_len, 0], true_abs_path[:obs_len, 1], 
                     'b.-', label='History')
            plt.plot(true_abs_path[obs_len-1:, 0], true_abs_path[obs_len-1:, 1], 
                     'g.-', label='Ground Truth')
            plt.plot(pred_abs[:, 0], pred_abs[:, 1], 
                     'r.--', label='Prediction')
            
            plt.scatter(start_pos[0], start_pos[1], c='k', marker='x')
            plt.title(f"Sample {count+1}")
            plt.grid(True, alpha=0.3)
            if count == 0: plt.legend()
            
            count += 1
            
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='行人轨迹预测模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='选择运行模式: train (训练并保存) 或 test (加载并测试)')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        test()