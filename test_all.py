import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from utils.dataset import PedestrianDataset
from utils.metrics import calculate_ade, calculate_fde
from models.mlp import TrajectoryMLP
from models.lstm import TrajectoryLSTM
from models.transformer import TrajectoryTransformer

def test_model(model_name):
    device = Config.DEVICE
    
    # 1. 加载测试集 (确保 phase='test' 以隔离训练数据)
    test_dataset = PedestrianDataset(
        Config.DATA_PATH, 
        Config.OBS_LEN, 
        Config.PRED_LEN, 
        phase='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. 初始化模型并加载权重
    if model_name == 'mlp':
        model = TrajectoryMLP()
    elif model_name == 'lstm':
        model = TrajectoryLSTM()
    elif model_name == 'transformer':
        model = TrajectoryTransformer()
    
    weight_path = os.path.join(Config.SAVE_DIR, f'{model_name}_model.pth')
    if not os.path.exists(weight_path):
        print(f"Warning: No weights found for {model_name} at {weight_path}")
        return None, None

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 在测试集上评估指标
    total_ade = 0
    total_fde = 0
    num_batches = len(test_loader)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_ade += calculate_ade(output, target)
            total_fde += calculate_fde(output, target)

    avg_ade = total_ade / num_batches
    avg_fde = total_fde / num_batches
    
    return avg_ade, avg_fde

def main():
    models_to_test = ['mlp', 'lstm', 'transformer']
    results = {}

    print(f"{'Model':<15} | {'ADE':<10} | {'FDE':<10}")
    print("-" * 40)

    for name in models_to_test:
        ade, fde = test_model(name)
        if ade is not None:
            results[name] = {'ADE': ade, 'FDE': fde}
            print(f"{name.upper():<15} | {ade:<10.4f} | {fde:<10.4f}")

    # 可选：将结果保存为 txt 报告
    if results:
        report_path = os.path.join('results', 'test_report.txt')
        with open(report_path, 'w') as f:
            f.write("Model Test Results (Test Set Only)\n")
            f.write("-" * 40 + "\n")
            for name, metrics in results.items():
                f.write(f"{name.upper()}: ADE={metrics['ADE']:.4f}, FDE={metrics['FDE']:.4f}\n")
        print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()