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
├── binning_report.txt      # 由优化器生成的数据分箱和聚类分析报告
└── sankey_report.html      # (新增) 由优化器生成的可视化桑基图报告
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

---
## 7. 模型设计与算法详解

本推荐系统的核心是一个基于**个性化PageRank (Personalized PageRank, PPR)** 算法的异构图谱模型。

### 7.1 图谱节点 (Input Nodes)
我们构建了一个包含6种不同类型节点的异构信息网络（Heterogeneous Information Network），用以捕捉市场中复杂的实体关系：
1.  **【承租人 (Lessee)】**: 资金需求方。
2.  **【出租人 (Lessor)】**: 资金提供方，也是我们最终要推荐的目标。
3.  **【省份 (Province)】**: 承租人所在的地理区域属性。
4.  **【行业 (Industry)】**: 承租人所属的行业属性（申万一级）。
5.  **【财产价值分箱 (Value Bin)】**: 为了将连续的财产价值离散化，我们使用分位数方法将其分为12个区间，每个区间为一个节点。
6.  **【租赁期限分箱 (Term Bin)】**: 类似地，我们将租赁期限通过K-Means聚类分为5个簇，每个簇为一个节点。

### 7.2 游走概率 (边权重计算)
图中的有向边权重代表了从一个节点“游走”到另一个节点的概率，其计算方式根据边的类型有所不同：

*   **属性节点 → 承租人**: 这类边的权重反映了该承租人在这个属性下的重要性。计算公式为：
    `权重 = (承租人A在属性X下的业务次数) / (属性X下的总业务次数)`
    例如，“山东省”节点到“承租人A”的权重，等于A在山东的业务笔数占山东总笔数的比例。

*   **承租人 → 出租人**: 这是模型的核心关系，其权重由一个综合分数决定，该分数由三个可调超参数控制：
    *   `w_freq`: **交易频率权重**，代表承租人与该出租人历史交易次数的占比。
    *   `w_val`: **交易金额权重**，代表承租人与该出租人历史交易总金额的占比。
    *   `lambda`: **时间衰减率**，用于给近期发生的交易赋予更高的权重，`exp(-lambda * days_ago)`。
    `optimizer.py` 脚本的核心任务就是通过回测数据，找到这三个超参数的最优组合。

*   **省份 ↔ 省份**: 为了解决部分省份数据稀疏的问题，我们通过分析各省的行业分布，使用K-Means将省份聚类。在同一个聚类簇中的省份，我们会根据其行业分布的**余弦相似度**，建立双向的、带权重的边。这使得信息可以在地理位置和经济结构相似的省份之间流动。

### 7.3 推荐列表生成 (Recommendation Generation)
当一个查询（例如：寻找为“山东省”的“农林牧渔”企业服务的出租人）输入时，推荐过程如下：
1.  **构建个性化向量**: 我们将查询中的核心属性节点（例如“山东省”节点、“农林牧渔”节点、以及查询金额与期限对应的分箱节点）作为PageRank算法的**个性化向量 (Personalization Vector)**。这意味着算法的“随机游走”将从这些节点以等概率起始。
2.  **运行PPR**: 以此个性化向量为输入，在全图上运行`networkx.pagerank`算法，得到图中每个节点的最终分数。

### 7.4 评分计算 (Scoring)
1.  **筛选与排序**: 在PPR运行结束后，我们只筛选出**【出租人】**类型的节点，并根据它们的PageRank分数从高到低进行排序。
2.  **分数归一化**: PageRank的原始分数比较抽象，为了使其更具业务含义，我们通过**Min-Max归一化**方法，将其线性映射到 **300-850** 的信用分区间，公式如下：
    `最终分数 = 300 + ((原始分数 - 最小原始分数) / (最大原始分数 - 最小原始分数)) * 550`

这样，我们就得到了最终带评分的Top-N推荐列表。

---
## 附录A: 在PyCharm中运行项目

对于使用PyCharm IDE的开发者，可以遵循以下步骤来配置和运行本项目。

### 步骤 1: 打开项目
- 启动PyCharm。
- 选择 `File -> Open...`，然后导航到并选中本项目的根文件夹。

### 步骤 2: 配置Python解释器
这是保证项目正常运行的最关键一步，目的是让PyCharm使用我们创建的虚拟环境。
- 打开设置: `File -> Settings...` (Windows/Linux) 或 `PyCharm -> Preferences...` (macOS)。
- 导航到 `Project: [你的项目名] -> Python Interpreter`。
- 点击解释器列表旁的齿轮图标，选择 `Add...`。
- 在弹出的窗口中，选择左侧的 `Existing environment`。
- 在 `Interpreter` 字段中，点击 `...` 按钮，找到并选中项目文件夹下的虚拟环境解释器。
  - 对于Windows: `venv\Scripts\python.exe`
  - 对于Mac/Linux: `venv/bin/python`
- 点击 `OK` 保存设置。PyCharm的底部状态栏会显示正在更新解释器，稍等片刻即可。

### 步骤 3: 安装依赖库
配置好解释器后，PyCharm通常会自动检测到 `requirements.txt` 文件，并在编辑器顶部显示一个提示条。
- **推荐方式**: 直接点击提示条中的 `Install requirements` 链接，PyCharm会自动完成安装。
- **备用方式**: 打开PyCharm底部的 `Terminal` 工具窗口，它会自动激活虚拟环境。在终端中手动运行命令 `pip install -r requirements.txt`。

### 步骤 4: 运行核心脚本
现在您可以方便地运行项目中的任意脚本了。
- **运行优化器**: 在左侧的项目文件浏览器中，找到 `optimizer.py`，右键点击它，然后选择 `Run 'optimizer'`。脚本的输出会显示在底部的 `Run` 工具窗口中。
- **启动Web应用**: 同样地，找到 `app.py`，右键点击并选择 `Run 'app'`。您会在 `Run` 工具窗口中看到Flask服务器启动的日志，包括其运行的地址（例如 `http://127.0.0.1:5001`）。
