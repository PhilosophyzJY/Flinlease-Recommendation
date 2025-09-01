# 融资租赁智能推荐系统 (专业版)

## 1. 项目目标

本项目旨在构建一个专业级的、可配置、可优化的智能推荐系统。它通过分析融资租赁市场中各参与方的复杂关系，为特定的承租人画像智能推荐最匹配的“出租人”。

该项目包含两个核心工作流：
1.  **模型优化 (`optimizer.py`)**: 一个自动化的超参数调优框架，通过历史数据回测来寻找最佳的模型参数。
2.  **推荐生成 (`recommender.py`)**: 一个使用优化后的最佳参数，在全量数据上运行并生成最终推荐列表的脚本。

## 2. 项目结构

```
.
├── finlease_train.csv      # 原始数据样本文件
├── config.ini              # 配置文件，管理路径和所有模型超参数
├── requirements.txt        # Python依赖库列表
├── utils.py                # 包含所有核心共享函数（数据处理、图谱构建等）的模块
├── optimizer.py            # 用于模型超参数优化的脚本
├── recommender.py          # 用于生成最终推荐的脚本
├── optimization_results.png # 由优化器生成的结果图表
└── binning_report.txt      # (新增) 由优化器生成的数据分箱和聚类分析报告
```

---

## 3. 环境搭建 (Environment Setup)

在运行此项目前，请确保您的电脑已安装 **Python 3**。

**步骤 1：创建虚拟环境 (推荐)**
```bash
python -m venv venv
```

**步骤 2：激活虚拟环境**
*   **Windows:** `venv\Scripts\activate`
*   **Mac/Linux:** `source venv/bin/activate`

**步骤 3：安装依赖库**
```bash
pip install -r requirements.txt
```
---

## 4. 如何使用 - 模式一：Web应用 (推荐)

这是推荐的、最用户友好的使用方式。

### 步骤 1: 运行优化器 (一次性)
首先，您需要运行优化器来生成数据处理规则和参数建议。这一步在您拿到新的数据集或者希望重新训练模型时运行。
```bash
python optimizer.py
```
运行结束后，请根据输出的建议和图表，**手动更新** `config.ini` 文件中的 `[Optimized_Parameters]` 部分。

### 步骤 2: 启动Web服务器
```bash
python app.py
```
服务器启动后，它会加载模型并持续在后台运行。

### 步骤 3: 在浏览器中访问
打开您的网页浏览器，访问以下地址：
**http://127.0.0.1:5001**

您现在可以通过网页界面来输入查询条件并获取实时的推荐结果。

---
## 6. 配置文件详解 (`config.ini`)

```ini
[Paths]
# --- 文件路径配置 ---
# 用于优化和评估模型的数据集
learning_data_file = finlease_train.csv
# 模型优化完成后, 最终用于生产推荐的数据集
production_data_file = finlease_train.csv
# 优化结果图表的保存路径
plot_file = optimization_results.png
# 训练阶段学习到的分箱规则的保存路径
binning_rules_file = binning_rules.pkl
# (新增) 由优化器生成的数据分箱和聚类分析报告的路径
report_file = binning_report.txt

[Data_Settings]
# --- 数据处理参数 ---
# K-Means对“租赁期限”分箱的数量, 建议范围: 3-8
n_term_bins = 5
# K-Means对“省份”分组的数量, 建议范围: 3-8
n_province_clusters = 5
# Quantile对“财产价值”分箱的数量
n_value_bins = 12

[Optimization_Settings]
# --- 优化器超参数配置 ---
# 定义训练集的年-月 (格式 YYYY-M, 可配置多个, 用逗号分隔)
train_year_months = 2025-5, 2025-6
# 定义测试集的年-月 (格式 YYYY-M, 可配置多个)
test_year_months = 2025-7

# 定义时间衰减率lambda的搜索范围
# 格式: 起始值, 结束值, 步数 (生成多少个候选值)
lambda_space = 0.001, 0.01, 5

# 定义承租人->出租人关系中，频率与金额权重的搜索范围
# w_freq的权重将在此范围搜索, w_val会自动设为(1 - w_freq)
# 格式: 起始值, 结束值, 步数
weight_space = 0.0, 1.0, 6

[Optimized_Parameters]
# !!! 注意: 请根据 optimizer.py 脚本运行后的建议结果, 在下方手动配置您最终选定的参数 !!!
# best_lambda: 关系强度的时间衰减率, 浮点数, e.g., 0.005
best_lambda = 0.001
# best_w_freq: Lessee -> Lessor 关系中, "交易频率"的权重, 浮点数, e.g., 0.4
best_w_freq = 0.2
# best_w_val: Lessee -> Lessor 关系中, "交易金额"的权重, 浮点数, e.g., 0.6
# 通常情况下, best_w_freq + best_w_val = 1.0
best_w_val = 0.8

[Learned_Rules_Summary]
# --- 以下部分由 optimizer.py 脚本自动生成, 仅供查阅 ---
# value_bin_boundaries = ...
# term_bin_centers_years = ...
# province_cluster_map_json = ...
```
