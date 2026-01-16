import torch
import matplotlib.pyplot as plt
import os
import random
import argparse
import numpy as np

from config import Config
from utils.dataset import PedestrianDataset
from models.mlp import TrajectoryMLP
from models.lstm import TrajectoryLSTM
from models.transformer import TrajectoryTransformer

def visualize_prediction(model, dataset, model_name, num_samples=3, save_dir='results'):
    """
    随机抽取样本并画出轨迹对比图
    """
    model.eval()
    device = Config.DEVICE
    
    # 创建保存路径，例如 results/mlp/vis/
    vis_save_path = os.path.join(save_dir, model_name, 'visualizations')
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    # 随机选择样本索引
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        obs, target = dataset[idx]
        
        # 准备输入数据 [1, obs_len, 2]
        obs_tensor = obs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(obs_tensor) # [1, pred_len, 2]
        
        # 转换回 CPU numpy 绘图
        obs_np = obs.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = pred.squeeze(0).cpu().numpy()
        
        # 开始绘图
        plt.figure(figsize=(8, 8))
        
        # 1. 历史轨迹 (蓝色)
        plt.plot(obs_np[:, 0], obs_np[:, 1], color='blue', marker='o', label='Observed (Past)', markersize=4)
        
        # 2. 真实未来 (绿色) - 连接观察的最后一点以保证线条连续
        true_future = np.vstack([obs_np[-1], target_np])
        plt.plot(true_future[:, 0], true_future[:, 1], color='green', marker='s', label='Ground Truth', markersize=4)
        
        # 3. 预测轨迹 (红色) - 连接观察的最后一点
        pred_future = np.vstack([obs_np[-1], pred_np])
        plt.plot(pred_future[:, 0], pred_future[:, 1], color='red', marker='x', linestyle='--', label='Prediction', markersize=4)
        
        plt.title(f"{model_name.upper()} Trajectory Prediction (Sample {idx})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 保存图片到对应算法的子目录下
        save_file = os.path.join(vis_save_path, f"sample_{idx}.png")
        plt.savefig(save_file, dpi=200)
        plt.close()
        print(f"Sample {idx} visualization saved to: {save_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm', 'transformer'])
    args = parser.parse_args()

    # 1. 加载数据
    test_dataset = PedestrianDataset(Config.DATA_PATH, Config.OBS_LEN, Config.PRED_LEN, phase='test')
    
    # 2. 初始化模型并加载权重
    if args.model == 'mlp':
        model = TrajectoryMLP()
    elif args.model == 'lstm':
        model = TrajectoryLSTM()
    elif args.model == 'transformer':
        model = TrajectoryTransformer()
        
    weight_path = os.path.join(Config.SAVE_DIR, f'{args.model}_model.pth')
    
    if not os.path.exists(weight_path):
        print(f"Error: Cannot find weights file {weight_path}.")
        return

    model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    print(f"Successfully loading {args.model} weights: {weight_path}")

    # 3. 执行可视化
    visualize_prediction(model, test_dataset, args.model, num_samples=5)

if __name__ == '__main__':
    main()