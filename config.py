import torch

class Config:
    # ---------------- 数据与路径 ----------------
    DATA_PATH = 'data/ewap_dataset/seq_eth/obsmat.txt'
    SAVE_DIR = 'saved_weights/'
    
    # ---------------- 轨迹参数 ----------------
    OBS_LEN = 8    # 观察过去 8 帧 (3.2秒)
    PRED_LEN = 12  # 预测未来 12 帧 (4.8秒)
    # 输入特征维度: (x, y, vx, vy) = 2
    INPUT_SIZE = 4 
    # 输出特征维度: (x, y) = 2
    OUTPUT_SIZE = 4
    
    # ---------------- 训练超参 ----------------
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000  # 【要求】相同的训练轮数
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------- 模型参数 (用于控制参数量) ----------------
    # MLP 配置
    MLP_HIDDEN_SIZE = 128
    
    # LSTM 配置
    LSTM_HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    
    # Transformer 配置
    TRANS_D_MODEL = 64
    TRANS_NHEAD = 4
    TRANS_LAYERS = 2
    TRANS_FF_DIM = 128
    TRANS_DROPOUT = 0.1