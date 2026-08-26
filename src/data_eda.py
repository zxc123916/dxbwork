import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 正确绝对路径
data_file = r"D:\githubdatadxb\dxbwork\data\dataset.xlsx"
docs_dir = r"D:\githubdatadxb\dxbwork\docs"

os.makedirs(docs_dir, exist_ok=True)

# 解决中文绘图乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

print("="*50)
print("📊 工业机器人健康状态监测 - 数据探索性分析")
print("="*50)

df = pd.read_excel(data_file, engine="openpyxl")
print(f"✅ 数据集读取成功！")
print(f"📐 数据集规模：{df.shape[0]} 行 × {df.shape[1]} 列")

print("\n📋 【列名与数据类型】")
print(df.dtypes)

print("\n⚠️ 【缺失值统计】")
missing_stats = df.isnull().sum()
print(missing_stats[missing_stats > 0])
print(f"💡 总缺失值数量：{missing_stats.sum()}")

print("\n📈 【数值字段描述性统计】")
print(df.describe())

print("\n🛡️  【保护停止标签(Robot_ProtectiveStop)分布】")
if "Robot_ProtectiveStop" in df.columns:
    print(df["Robot_ProtectiveStop"].value_counts())
else:
    print("❌ 未找到 Robot_ProtectiveStop 列")

print("\n🤝 【抓手丢失标签(grip_lost)分布】")
if "grip_lost" in df.columns:
    print(df["grip_lost"].value_counts())
else:
    print("❌ 未找到 grip_lost 列")

print("\n🎨 正在生成特征相关性热力图...")
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("工业机器人传感器特征相关性热力图", fontsize=14)
plt.tight_layout()

heatmap_path = os.path.join(docs_dir, "特征相关性热力图.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 热力图已保存至：{heatmap_path}")

print("\n" + "="*50)
print("🎉 数据探索性分析全部完成！")
print("="*50)