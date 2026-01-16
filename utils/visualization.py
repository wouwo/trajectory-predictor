import matplotlib.pyplot as plt
import os

def plot_training_curves(loss_history, ade_history, fde_history, model_name, save_dir='results'):
    """
    绘制训练曲线
    1. 从第 10 轮开始绘制，避开初期的不稳定阶段
    2. 每 10 轮绘制一个点，使长周期训练（如 1000 轮）的图表更清晰
    """
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置绘图间隔和起始轮次
    interval = 10 

    metrics = [
        {'data': loss_history, 'name': 'Loss', 'color': 'blue', 'ylabel': 'Loss (MSE)'},
        {'data': ade_history, 'name': 'ADE', 'color': 'orange', 'ylabel': 'ADE (meters/pixels)'},
        {'data': fde_history, 'name': 'FDE', 'color': 'green', 'ylabel': 'FDE (meters/pixels)'}
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        
        # --- 核心逻辑修改 ---
        # 1. 从第 10 轮（索引为 9）开始取数据，步长为 10
        # 如果 loss_history[0] 是第一轮的结果，那么 loss_history[9] 就是第十轮
        plot_data = metric['data'][interval-1::interval]
        
        # 2. 生成对应的 X 轴坐标 (10, 20, 30, ..., NUM_EPOCHS)
        # 确保坐标点与数据点在数量上完全匹配
        epochs = range(interval, len(metric['data']) + 1, interval)
        
        # 3. 绘图：添加 marker='o' 以增强数据点的可视性
        plt.plot(epochs, plot_data, label=f"Train {metric['name']}", 
                 color=metric['color'], marker='o', markersize=1, linestyle='-')
        # --------------------

        plt.title(f"{model_name.upper()} Training {metric['name']} (From Epoch {interval})")
        plt.xlabel('Epoch')
        plt.ylabel(metric['ylabel'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        filename = f"{model_name}_{metric['name'].lower()}_curve.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    print(f"Metrics curves (starting from Epoch {interval}) saved to {save_dir}/")