# 工业机器人健康状态监测系统
课程设计项目：基于CNN‑LSTM的工业机器人故障健康状态监测，实现时序传感器数据故障识别，配套FastAPI网页推理系统。

## 项目说明
本项目针对工业机器人传感器时序数据开展健康状态监测，采用CNN‑LSTM混合深度学习网络，CNN层完成局部特征提取，LSTM层学习时序依赖关系，实现机器人故障二分类识别（0‑健康，1‑故障，需要维护）。
提供Web网页交互界面，可以输入传感器数据完成在线推理预测。

## 目录结构
dxbwork
├─src
│ ├─api_server.py # FastAPI 后端服务主程序
│ ├─train_cnn_lstm.py # CNN‑LSTM 模型训练脚本
│ └─model_infer.py # 模型推理模块
├─templates
│ ├─input.html # 数据输入网页
│ └─result.html # 推理结果展示网页
├─db # 数据库存放目录
├─dataset.xlsx # 原始数据集（项目根目录）
├─requirements.txt # 项目依赖包
└─README.md # 项目说明文档

## 数据来源
地址：https://archive.ics.uci.edu/dataset/963/ur3%2Bcobotops#1
数据集文件：`dataset.xlsx`，放置项目根目录。
数据集包含工业机器人多维度传感器时序采集数据，总共7409条样本，共24列特征；
标签字段：`grip_lost`，布尔类型，`True`代表机器人发生故障，`False`代表设备正常运行。
数据集存在样本不均衡：正常样本7166条，故障样本243条。

## ⚙️数据预处理说明
预处理脚本：`src/data_preprocess.py`
1. **无效列过滤**：剔除原始数据中非数值类型时间字符串列，仅保留全部传感器数值特征。
2. **缺失值处理**：对数据中少量缺失值采用均值填充，避免空值造成训练报错。
3. **特征标准化**：对全部传感器特征做标准化处理，消除量纲差异，标准化参数保存为`model/scaler.pkl`。
4. **输出保存**：预处理完成数据保存为`data/processed_data.csv`，可直接用于模型训练。
5. **时序样本构造**：训练脚本`train_cnn_lstm.py`读取`processed_data.csv`，使用滑动窗口切分，构建CNN‑LSTM所需时序样本。
6. **数据集划分**：按照7:3划分训练集、测试集，采用分层划分保证训练集与测试集故障样本比例一致。
7. **损失权重处理**：数据集正负样本不均衡，训练使用加权交叉熵损失函数，提高故障类别损失权重，提升故障样本召回率。
8. **模型评估**：训练完成使用混淆矩阵、召回率、准确率对模型综合评估，重点关注故障类别识别效果。


数据预处理完
## 环境依赖
fastapi
uvicorn
pandas
numpy
torch
scikit‑learn
python‑multipart
jinja2
openpyxl

## 模型训练
python src/train_cnn_lstm.py

## 启动网页后端服务
python src/api_server.py

http://127.0.0.1:8000