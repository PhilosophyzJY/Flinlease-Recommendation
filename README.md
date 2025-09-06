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
└── bi_report.html          # (新增) 由优化器生成的可视化商业智能报告
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

# 定义属性权重组合的搜索步长。例如, 0.25 会测试 (1,0,0,0), (0.75,0.25,0,0) 等组合
attribute_weight_step = 0.25


[Optimized_Parameters]
# !!! 注意: 请根据 optimizer.py 脚本运行后的建议结果, 在下方手动配置您最终选定的参数 !!!
# best_lambda: 关系强度的时间衰减率
best_lambda = 0.001
# best_w_province: 省份属性的权重
best_w_province = 0.25
# best_w_industry: 行业属性的权重
best_w_industry = 0.50
# best_w_value: 财产价值属性的权重
best_w_value = 0.25
# best_w_term: 租赁期限属性的权重
best_w_term = 0.00

[Learned_Rules_Summary]
# --- 以下部分由 optimizer.py 脚本自动生成, 仅供查阅 ---
# value_bin_boundaries = ...
# term_bin_centers_years = ...
# province_cluster_map_json = ...
```

---
## 7. 数据预处理与特征工程 (Data Preprocessing & Feature Engineering)

在构建图谱和模型之前，我们对原始数据进行了关键的预处理和特征工程，旨在将原始的、多样的交易记录转化为模型可以高效利用的结构化特征。这些步骤对于模型的最终性能至关重要。

### 7.1 核心思想：离散化与降维

面对连续的数值（如财产价值）和高基数的分类变量（如省份），我们的核心策略是**离散化 (Discretization)** 和 **降维 (Dimensionality Reduction)**。
*   **离散化**: 将连续变量或多值变量转化为有限的、可管理的类别。这有助于模型捕捉非线性关系，并降低异常值（如极高或极低的财产价值）的直接影响。
*   **降维**: 将具有相似行为的实体（如经济结构相似的省份）分组。这解决了数据稀疏性问题，并使模型能够泛化，发现更高层次的模式。

### 7.2 具体实现

#### 1. **财产价值分箱 (Value Binning)**
*   **目的**: 将连续的“财产价值”转化为有序的类别，如“低价值”、“中等价值”、“高价值”等。
*   **方法**: 我们采用**分位数分箱 (Quantile Binning)**。该方法将所有交易按价值排序，然后将它们平分到 `n_value_bins` 个桶中，确保每个桶包含大致相同数量的交易。
*   **优势**: 相比于等距分箱，分位数分箱能更好地处理数据倾斜（即大部分交易集中在某个价值区间）的情况，保证了每个“价值”节点的下游数据量相对均衡。

#### 2. **租赁期限分箱 (Term Binning)**
*   **目的**: 将具体的租赁期限（如36个月、60个月）聚合成有意义的类别（如“短期”、“中期”、“长期”）。
*   **方法**: 我们使用 **K-Means 聚类算法**。该算法将所有“租赁期限”数值视为一维空间中的点，并寻找 `n_term_bins` 个聚类中心，使得每个点都属于离它最近的中心。
*   **优势**: K-Means能够自动发现数据中自然的期限聚集模式，比手动设定阈值更客观、更数据驱动。

#### 3. **省份聚类 (Province Clustering)**
*   **目的**: 解决数据稀疏性问题。对于交易较少的省份，直接为其推荐可能很困难。通过将经济结构相似的省份聚类，模型可以利用整个聚类的集体智慧为稀疏省份进行推荐。
*   **方法**:
    1.  **构建省份画像**: 我们首先为每个省份创建一个“行业画像”向量，向量的每一维代表一个行业，其值是该行业在该省份的交易总额。
    2.  **K-Means聚类**: 我们对这些画像向量运行 **K-Means 聚类算法**，将具有相似行业分布的省份分到同一个簇中。
*   **结果**: 这不仅是一种特征工程，它还直接影响了图的结构。在后续的图构建中，同一个簇内的省份之间会根据其画像的**余弦相似度**被添加带权重的边，使得信息可以在它们之间有效流动。

---

## 8. 模型设计与算法详解

本推荐系统的核心是一个基于**个性化PageRank (Personalized PageRank, PPR)** 算法的异构图谱模型。

### 8.1 图谱节点 (Input Nodes)
我们构建了一个包含6种不同类型节点的异构信息网络（Heterogeneous Information Network），用以捕捉市场中复杂的实体关系：
1.  **【承租人 (Lessee)】**: 资金需求方。
2.  **【出租人 (Lessor)】**: 资金提供方，也是我们最终要推荐的目标。
3.  **【省份 (Province)】**: 承租人所在的地理区域属性。
4.  **【行业 (Industry)】**: 承租人所属的行业属性（申万一级）。
5.  **【财产价值分箱 (Value Bin)】**: 为了将连续的财产价值离散化，我们使用分位数方法将其分为12个区间，每个区间为一个节点。
6.  **【租赁期限分箱 (Term Bin)】**: 类似地，我们将租赁期限通过K-Means聚类分为5个簇，每个簇为一个节点。

### 8.2 游走概率 (边权重计算)
图中的有向边权重代表了从一个节点“游走”到另一个节点的概率。该算法的核心是为每笔交易计算一个最终的`final_score`，所有边的权重都基于这个分数进行归一化。

`final_score`的计算过程如下：

**第一步：计算特定上下文的平均期限 (`avg_term`)**
为了更精细地捕捉时间价值，我们认为交易的“合理”期限受其属性（如省份、行业）和自身价值的影响。因此，我们为每个交易在四个不同上下文中计算平均期限：
*   `avg_term_province` = 特定省份下、特定价值分箱内的所有交易的平均期限。
*   `avg_term_industry` = 特定行业下、特定价值分箱内的所有交易的平均期限。
*   `avg_term_value` = 特定价值分箱内的所有交易的平均期限。
*   `avg_term_term` = 特定期限分箱下、特定价值分箱内的所有交易的平均期限。

**第二步：计算特定上下文的复合分数 (`composite_score`)**
我们结合交易价值、平均期限和时间衰减，为每个上下文计算一个分数：
`composite_score_context = 财产价值 * exp(avg_term_context - lambda * 交易距今的天数)`

**第三步：计算最终加权分数 (`final_score`)**
我们将四个上下文的分数根据其重要性（权重）进行加权求和，得到最终分数。这些权重是模型需要优化的核心参数。
`final_score = (w_prov * score_prov) + (w_ind * score_ind) + (w_val * score_val) + (w_term * score_term)`

*   `lambda`, `w_prov`, `w_ind`, `w_val`, `w_term` 都是通过 `optimizer.py` 脚本在历史数据上寻找出的最优超参数。

**边的权重计算**
*   **属性节点 → 承租人**: `权重 = (承租人A在属性X的所有交易的final_score总和) / (属性X内所有交易的final_score总和)`
*   **承租人 → 出租人**: `权重 = (承租人A与出租人Y之间所有交易的final_score总和) / (承租人A所有交易的final_score总和)`

*   **省份 ↔ 省份**: (无变化) 为了解决部分省份数据稀疏的问题，我们通过分析各省的行业分布，使用K-Means将省份聚类。在同一个聚类簇中的省份，我们会根据其行业分布的**余弦相似度**，建立双向的、带权重的边。

### 8.3 推荐列表生成 (Recommendation Generation)
当一个查询（例如：寻找为“山东省”的“农林牧渔”企业服务的出租人）输入时，推荐过程如下：
1.  **构建个性化向量**: 我们将查询中的核心属性节点（例如“山东省”节点、“农林牧渔”节点等）作为PageRank算法的**个性化向量 (Personalization Vector)**。随机游走的起始概率将根据优化后的属性权重（`w_prov`, `w_ind`等）进行分配，而不是平均分配。
2.  **运行PPR**: 以此个性化向量为输入，在全图上运行`networkx.pagerank`算法，得到图中每个节点的最终分数。

### 8.4 评分计算 (Scoring)
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
