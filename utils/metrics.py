import torch

def calculate_ade(pred, target):
    """
    计算 Average Displacement Error (ADE)
    Args:
        pred: 预测轨迹 [batch_size, pred_len, 2]
        target: 真实轨迹 [batch_size, pred_len, 2]
    Returns:
        avg_ade: 标量
    """
    # 计算每个时间步的欧氏距离: sqrt((x-x_gt)^2 + (y-y_gt)^2)
    displacement = torch.norm(pred[:, :, :2] - target[:, :, :2], dim=2) 
    # 对所有步长、所有样本求平均
    avg_ade = torch.mean(displacement)
    return avg_ade.item()

def calculate_fde(pred, target):
    """
    计算 Final Displacement Error (FDE)
    只关注最后一个时间步的误差
    Args:
        pred: 预测轨迹 [batch_size, pred_len, 2]
        target: 真实轨迹 [batch_size, pred_len, 2]
    Returns:
        avg_fde: 标量
    """
    # 取最后一个时间步: shape [batch_size, 2]
    final_pred = pred[:, -1, :]
    final_target = target[:, -1, :]
    
    # 计算欧氏距离
    displacement = torch.norm(final_pred[:, :2] - final_target[:, :2], dim=1)
    # 对所有样本求平均
    avg_fde = torch.mean(displacement)
    return avg_fde.item()