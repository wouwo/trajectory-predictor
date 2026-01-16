import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from config import Config
from utils.dataset import PedestrianDataset
from utils.metrics import calculate_ade, calculate_fde
from utils.visualization import plot_training_curves
from models.mlp import TrajectoryMLP
from models.lstm import TrajectoryLSTM
from models.transformer import TrajectoryTransformer

# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    # 选择模型['mlp','lstm','transformer']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm', 'transformer'])
    args = parser.parse_args()

    # 1. 加载三个数据集
    train_dataset = PedestrianDataset(Config.DATA_PATH, Config.OBS_LEN, Config.PRED_LEN, phase='train')
    val_dataset = PedestrianDataset(Config.DATA_PATH, Config.OBS_LEN, Config.PRED_LEN, phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    loss_history = []
    ade_history = []
    fde_history = []

    if args.model == 'mlp':
        model = TrajectoryMLP()
    elif args.model == 'lstm':
        model = TrajectoryLSTM()
    elif args.model == 'transformer':
        model = TrajectoryTransformer()
    
    model = model.to(Config.DEVICE)
    print(f"[{args.model.upper()}] Params: {count_parameters(model)}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3)

    for epoch in range(Config.NUM_EPOCHS):
        # --- 训练逻辑 ---
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        model.eval()
        total_val_ade = 0
        total_val_fde = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
                output = model(data)
                total_val_ade += calculate_ade(output, target)
                total_val_fde += calculate_fde(output, target)
        
        avg_val_ade = total_val_ade / len(val_loader)
        avg_val_fde = total_val_fde / len(val_loader)

        ade_history.append(avg_val_ade)
        fde_history.append(avg_val_fde)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | Loss: {avg_loss:.4f} | ADE: {avg_val_ade:.4f} | FDE: {avg_val_fde:.4f}')

    # 3. 训练结束后，保存最后的模型权重
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f'{args.model}_model.pth'))

    plot_training_curves(loss_history, ade_history, fde_history, args.model, save_dir='results')
if __name__ == '__main__':
    train()