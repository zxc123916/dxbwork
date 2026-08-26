import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 简易占位模型（演示用，不需要加载训练权重）
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.linear = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h,_) = self.lstm(x)
        out = self.linear(h[-1])
        return self.sigmoid(out)

# 初始化模型（随机权重，仅页面演示）
model = SimpleLSTM(input_dim=24)
model.eval()

def model_predict(df: pd.DataFrame):
    """
    对外推理接口，api_server调用这个函数
    :param df: DataFrame 上传的表格数据
    :return: (pred_label:int, pred_prob:float)
    """
    # 简单取数值列
    arr = df.select_dtypes(include=[np.number]).values
    # 补齐维度 (batch, seq_len, feature)
    tensor_x = torch.from_numpy(arr).float().unsqueeze(0)
    with torch.no_grad():
        prob = model(tensor_x)
    pred_prob = float(prob.squeeze().cpu().numpy())
    pred_label = 1 if pred_prob>0.5 else 0
    return pred_label, pred_prob