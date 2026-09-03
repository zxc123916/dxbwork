# 工业机器人健康状态监测系统
课程设计项目：基于CNN‑LSTM的工业机器人故障健康状态监测，实现时序传感器数据故障识别，配套FastAPI网页推理系统。

## 项目说明
本项目针对工业机器人传感器时序数据开展健康状态监测，采用CNN‑LSTM混合深度学习网络，CNN层完成局部特征提取，LSTM层学习时序依赖关系，实现机器人故障二分类识别（0‑健康，1‑故障，需要维护）。
提供Web网页交互界面，可以输入传感器数据完成在线推理预测。

## 目录结构
dxbwork
├── data                     # 数据集目录
│   ├── dataset.xlsx         # 原始数据集
│   ├── processed_data.csv   # 预处理完成数据集
│   └── 数据来源.txt
├── db                       # sqlite数据库存储检测历史记录
│   └── robot_fault.db
├── docs                     # 文档、实验图表
├── fig                      # 绘图输出图片
├── model                    # 训练产出模型文件
│   ├── cnn_lstm_best.pt     # 训练保存权重
│   └── scaler.pkl           # 标准化转换器
├── prompt                   # 对话记录（课程设计附录素材）
├── src                      # 后端Python源码
│   ├── api_server.py        # FastAPI网页后端主程序
│   ├── data_eda.py          # 数据探索分析
│   ├── data_preprocess.py   # 数据预处理脚本
│   ├── model_infer.py       # 模型推理封装
│   └── train_cnn_lstm.py    # CNN‑LSTM模型训练脚本
├── templates                # 前端网页模板
│   ├── input.html           # 数据输入页面
│   └── result.html          # 检测结果页面
├── 方案设计.md
├── 选题说明.md
├── 学习笔记.md
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明文档


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


网页系统功能
输入页面：两组传感器样本输入，每组 6 个 0‑0.4 浮点数；支持一键随机生成数据、清空全部输入；输入框实时校验数字与数值范围。
结果页面：展示故障标签、故障概率、推理耗时、风险等级、维护建议。
导出功能：保存本次检测结果，点击按钮下载 txt 格式实验报告，保存传感器原始数据、时间、推理指标。
持久化存储：SQLite 数据库自动保存每一次检测历史记录。
注意：修改数据表结构后，请删除db/robot_fault.db旧数据库文件，重启服务自动生成全新数据表。
停止服务快捷键：Ctrl + C