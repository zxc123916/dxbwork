import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import matplotlib.pyplot as plt
import seaborn as plt_sns

# ===================== 超参数配置 =====================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 3e-4
WINDOW_SIZE = 10
GRAD_CLIP = 1.0        # 梯度裁剪，防止梯度爆炸loss=nan
# 加权交叉熵权重：故障类权重设为35，正常类1，错分故障样本会得到35倍的损失惩罚
CLASS_WEIGHT = torch.tensor([1.0, 35.0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"D:\githubdatadxb\dxbwork\data\dataset.xlsx"
MODEL_SAVE_PATH = r"D:\githubdatadxb\dxbwork\model\cnn_lstm_best.pt"
FIG_SAVE_PATH = r"D:\githubdatadxb\dxbwork\fig\confusion_matrix_cnnlstm.png"

np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 数据读取与预处理 =====================
df = pd.read_excel(DATA_PATH)
print(f"数据集 size:{df.shape}")
print("标签分布：")
print(df["grip_lost"].value_counts())

# 只保留数值型传感器特征列
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != "grip_lost"]
X_raw = df[feature_cols].values.astype(np.float32)
y_all = df["grip_lost"].astype(int).values

# 特征标准化，解决数值尺度差异导致的训练不稳定
scaler = StandardScaler()
X_all = scaler.fit_transform(X_raw)

# 滑动窗口构造时序样本（给LSTM前后时序信息）
def create_sliding_window(x_arr, y_arr, win):
    xs, ys = [], []
    for i in range(len(x_arr)-win):
        xs.append(x_arr[i:i+win,:])
        ys.append(y_arr[i+win-1]) # 窗口最后时刻的故障标签
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sliding_window(X_all, y_all, WINDOW_SIZE)
print(f"滑动窗口后样本数：{X_seq.shape[0]}, 时序长度：{X_seq.shape[1]}, 特征数：{X_seq.shape[2]}")

# 分层划分训练/测试集，保证两类样本比例一致
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_seq, y_seq, test_size=0.3, random_state=SEED, stratify=y_seq
)

# ===================== 数据集与数据加载器 =====================
class RobotDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.from_numpy(data_x)
        self.y = torch.from_numpy(data_y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = RobotDataset(X_train_np, y_train_np)
test_ds = RobotDataset(X_test_np, y_test_np)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===================== CNN-LSTM 模型 =====================
class CNNLSTM(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # CNN模块：卷积层→激励层→池化层
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM模块
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        # 输出层
        self.fc_out = nn.Linear(64, 2)

    def forward(self, x):
        # 输入形状 (B, seq_len, feature_dim)
        x = x.permute(0,2,1) # 适配Conv1d输入 (B, feature_dim, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0,2,1) # 转回LSTM输入 (B, seq_len, feature_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:,-1,:] # 取最后一个时间步输出
        last_hidden = self.dropout(last_hidden)
        out = self.fc_out(last_hidden)
        return out

n_feat = X_seq.shape[-1]
model = CNNLSTM(n_feat).to(DEVICE)

# 加权交叉熵损失，故障类权重35，重点惩罚故障样本错分
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHT.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=12, gamma=0.65) # 学习率衰减，后期稳定收敛

# ===================== 训练&验证函数 =====================
def run_epoch(loader, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_pred = []
    all_true = []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) # 梯度裁剪防爆炸
                optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits.detach().cpu(), dim=1).numpy()
            y_cpu = y.detach().cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y_cpu)

    avg_loss = total_loss / len(loader)
    # 计算故障召回率（核心评判指标）
    recall = recall_score(all_true, all_pred, pos_label=1, zero_division=0)
    return avg_loss, all_true, all_pred, recall

# ===================== 主训练循环 =====================
best_recall = 0.0  # 以故障召回率为核心保存最优模型
print("===== CNN-LSTM 训练（加权交叉熵+召回率优先） =====")
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_y_true, tr_y_pred, tr_recall = run_epoch(train_loader, is_train=True)
    te_loss, te_y_true, te_y_pred, te_recall = run_epoch(test_loader, is_train=False)
    scheduler.step()

    print(f"Epoch:{epoch:2d} | Train loss:{tr_loss:.4f} | 训练召回率:{tr_recall:.4f} | Test loss:{te_loss:.4f} | 测试召回率:{te_recall:.4f}")

    # 按故障召回率保存最优模型
    if te_recall > best_recall:
        best_recall = te_recall
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  >> 保存最优模型（当前最高故障召回率：{best_recall:.4f}）")

# ===================== 最终评估：混淆矩阵+召回率核心评判 =====================
print("\n========== 最终测试集评估结果 ==========")
print(f"最高故障召回率：{best_recall:.4f}")
cm = confusion_matrix(te_y_true, te_y_pred)
print("\n混淆矩阵：")
print(cm)
print("\n分类报告（重点关注故障类召回率）：")
print(classification_report(te_y_true, te_y_pred, target_names=["正常(无需维护)","故障(需要维护)"], zero_division=0))

# 绘制混淆矩阵
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,5))
plt_sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["正常","故障"],
            yticklabels=["正常","故障"])
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("CNN-LSTM 混淆矩阵")
plt.tight_layout()
plt.savefig(FIG_SAVE_PATH, dpi=150)
plt.show()

print("\n说明：")
print("0 → 正常，机器人无需维护")
print("1 → 故障，机器人需要维护")