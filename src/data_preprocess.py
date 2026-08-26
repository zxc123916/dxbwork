import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

raw_file = r"D:\githubdatadxb\dxbwork\data\dataset.xlsx"
save_file = r"D:\githubdatadxb\dxbwork\data\processed_data.csv"

# 1.读取原始数据
df = pd.read_excel(raw_file, engine="openpyxl")
print(f"✅ 原始数据读取完成：{df.shape}")

# 2.缺失值：数值列中位数填充
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("✅ 缺失值填充完毕")

# 3.IQR改为clip截断（关键！不删除样本，限制数值上下限）
def iqr_clip(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower, upper)
    return data

feature_cols = [c for c in num_cols if c not in ["Robot_ProtectiveStop", "grip_lost"]]
for col in feature_cols:
    df = iqr_clip(df, col)
print(f"✅ IQR clip截断完成，样本数不变：{df.shape}")

# 4.Z‑score标准化
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
print("✅ Z‑score标准化完成")

# 保存，这里**不再做数据集划分**，划分放到训练脚本里做
df.to_csv(save_file, index=False, encoding="utf-8-sig")
print(f"✅ 预处理数据保存至 {save_file}")