# 行人轨迹预测

本项目是一个基于 PyTorch 实现的行人轨迹预测框架，支持多种主流深度学习模型，并集成了完整的训练、验证及对比流程。

## 1. 环境配置

推荐使用 **Python 3.8+** 环境，安装以下核心依赖库：

```bash
pip install torch matplotlib numpy pandas
```

## 2. 项目结构

```Plaintext
├── data/               # ETH行人轨迹预测数据集
├── models/             # 模型定义 (MLP, LSTM, Transformer)
├── utils/              # 工具类
│   ├── dataset.py      # 数据加载与预处理
│   ├── metrics.py      # 评价指标计算 (ADE, FDE)
│   └── visualization.py # 绘图与可视化工具
├── config.py           # 核心超参数配置 (Learning Rate, Batch Size, etc.)
├── train.py            # 训练脚本
├── test_all.py         # 一键化多模型性能对比测试
├── save_weights/       # 自动创建：存放训练的最佳权重 (.pth)
└── results/            # 自动创建：存放生成的训练指标曲线图
```

## 3. 使用说明

### 训练模型

通过`--model`参数指定架构。

```Bash
# 分别训练不同架构的模型
python train.py --model mlp
python train.py --model lstm
python train.py --model transformer
```

### 性能对比

训练完成后，运行测试脚本即可在测试集上对比所有模型的真实表现：

```Bash
python test_all.py
```

| Model   | ADE(平均距离误差) |     FDE(最终距离误差) |
| :--- | :--: | -------: |
| MLP |  0.5997  | 0.9368 |
| LSTM |  0.5618  | 0.9129 |
| Transformer |  0.4927  | 0.8527 |
