# 融资租赁智能推荐系统 (专业版)

## 1. 项目目标

本项目旨在构建一个专业级的、可配置、可优化的智能推荐系统。它通过分析融资租赁市场中各参与方的复杂关系，为特定的承租人画像智能推荐最匹配的“出租人”。

项目最终交付为一套Python脚本，核心功能包括：
*   **模型优化 (`optimizer.py`)**: 自动化的超参数调优框架。
*   **推荐生成 (`recommender.py`)**: 使用优化后的参数生成最终推荐。

---

## 2. 快速上手指南 (Quick Start Guide)

请按照以下三个核心步骤，快速运行本项目。

### 步骤 1: 安装依赖

在您的终端中，运行以下命令：
```bash
pip install -r requirements.txt
```

### 步骤 2: (可选) 优化模型参数

运行优化器脚本来为您的数据集寻找最佳参数。
```bash
python optimizer.py
```
*   这个过程会花费一些时间，因为它在测试多种参数组合。
*   运行结束后，它会自动将找到的**最优参数**更新到 `config.ini` 文件中。
*   同时，它会生成一张名为 `optimization_results.png` 的图表，您可以打开它来直观地查看调优结果。

### 步骤 3: 获取推荐结果

运行推荐器脚本，它将使用 `config.ini` 中的最优参数来计算。
```bash
python recommender.py
```
*   脚本会加载全部数据，使用优化后的参数构建模型，并为一个样本查询打印出Top 10推荐列表。

---

## 3. 项目配置 (Configuration)

您可以直接修改 `config.ini` 文件来调整项目的行为。

*   **`[Paths]`**:
    *   `data_file`: 指定您要使用的原始数据CSV文件名。
    *   `plot_file`: 指定优化结果图表的保存路径。
*   **`[Data]`**:
    *   `n_term_bins`: K-Means对“租赁期限”分箱的数量。
    *   `n_province_clusters`: K-Means对“省份”分组的数量。
*   **`[Tuning]`**:
    *   `lambda_space`: 定义时间衰减率的搜索范围 `(起始值, 结束值, 步数)`。
    *   `weight_space`: 定义承租人->出租人关系中，频率与金额权重的搜索范围 `(起始值, 结束值, 步数)`。
*   **`[Optimized_Parameters]`**:
    *   此部分由 `optimizer.py` **自动生成和更新**，无需手动修改。`recommender.py` 会读取这里的参数。

---

## 4. 模型核心逻辑详解

本项目最终采用的是一个统一的、信息更丰富的**异构图模型 (Heterogeneous Graph)**。其核心工作原理请参考 `utils.py` 脚本中的详细注释，以及`optimizer.py`和`recommender.py`的执行逻辑。
