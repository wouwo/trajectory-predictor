import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PedestrianDataset(Dataset):
    def __init__(self, data_path, obs_len, pred_len, phase='train'):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        
        # 加载全部数据
        all_inputs, all_targets = self.load_data(data_path)
        num_samples = len(all_inputs)
        
        # 按照 7:1:2 划分
        train_end = int(num_samples * 0.7)
        val_end = int(num_samples * 0.8) # 0.7 + 0.1
        
        if phase == 'train':
            self.inputs = all_inputs[:train_end]
            self.targets = all_targets[:train_end]
        elif phase == 'val':
            self.inputs = all_inputs[train_end:val_end]
            self.targets = all_targets[train_end:val_end]
        elif phase == 'test':
            self.inputs = all_inputs[val_end:]
            self.targets = all_targets[val_end:]

    def load_data(self, file_path):
        try:
            all_data = np.loadtxt(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

        inputs, targets = [], []
        
        # 获取所有唯一的行人 ID
        ped_ids = np.unique(all_data[:, 1])
        
        for ped_id in ped_ids:
            # 提取该行人的所有数据
            ped_data = all_data[all_data[:, 1] == ped_id]
            
            # 按帧号排序
            ped_data = ped_data[ped_data[:, 0].argsort()]

            coords = ped_data[:, [2, 4, 5, 7]] 
            
            # 滑动窗口构建样本
            num_samples = len(coords) - self.seq_len + 1
            
            for i in range(num_samples):
                # 输入: 0 到 obs_len
                inputs.append(coords[i : i + self.obs_len])
                # 标签: obs_len 到 obs_len + pred_len
                targets.append(coords[i + self.obs_len : i + self.seq_len])
                
        self.inputs = torch.from_numpy(np.array(inputs)).float()
        self.targets = torch.from_numpy(np.array(targets)).float()
        
        # print(f"Loaded {len(self.inputs)} samples from {file_path}")
        return self.inputs, self.targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]