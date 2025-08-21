# 融资租赁智能推荐系统

这是一个基于PageRank算法的智能推荐系统，旨在根据承租人的特定需求（如所在省份、所属行业），为其推荐最合适的融资租赁出租人。

该系统采用“计算与服务分离”的架构，包括一个用于模型训练的脚本和一个用于提供Web服务的轻量级应用。

## 项目结构

```
.
├── finlease_train.csv      # 原始数据样本文件
├── train.py                # 用于模型训练和生成数据产物的脚本
├── app.py                  # 用于提供Web服务的Flask应用
├── data.pkl                # 经过处理和分箱后的数据（由train.py生成）
├── graph.gpickle           # 计算好的图谱网络模型（由train.py生成）
├── data_processing.py      # 数据加载和清洗模块
├── feature_engineering.py  # 特征工程（价值分箱）模块
├── graph_builder.py        # 图谱构建模块
├── recommender.py          # PageRank推荐算法模块
├── templates/
│   └── index.html          # 前端页面
└── requirements.txt        # Python依赖库列表
```

---

## 1. 环境搭建 (Environment Setup)

在运行此项目前，请确保您的电脑已安装 **Python 3**。

**步骤 1：创建虚拟环境 (推荐)**

为了保持项目依赖的独立性，建议您创建一个虚拟环境。打开您的终端（在Windows上是命令提示符或PowerShell，在Mac上是终端）。

```bash
# 进入项目根目录
cd path/to/your/project

# 创建一个名为 venv 的虚拟环境
python -m venv venv
```

**步骤 2：激活虚拟环境**

*   **在 Windows 上:**
    ```bash
    venv\Scripts\activate
    ```

*   **在 Mac 或 Linux 上:**
    ```bash
    source venv/bin/activate
    ```
    激活后，您应该会在终端提示符前看到 `(venv)` 字样。

**步骤 3：安装依赖库**

在激活的虚拟环境中，运行以下命令来安装所有必需的Python库：

```bash
pip install -r requirements.txt
```

至此，您的本地环境已准备就绪。

---

## 2. 如何使用 (How to Use)

请按照以下步骤来更新数据、训练模型和启动应用。

**步骤 1：更新数据 (Update Data)**

1.  将您自己的融资租赁数据CSV文件放置在项目的根目录下。**请确保您的数据格式与 `finlease_train.csv` 的模板一致**。
2.  用文本编辑器打开 `train.py` 文件。
3.  在文件顶部的配置区，修改 `INPUT_DATA_FILE` 变量的值，使其指向您自己的文件名。
    ```python
    # --- Configuration ---
    INPUT_DATA_FILE = 'your_new_data.csv' # <--- 修改这里
    OUTPUT_DATA_PKL = 'data.pkl'
    OUTPUT_GRAPH_PKL = 'graph.gpickle'
    ```

**步骤 2：训练模型 (Train the Model)**

在您的终端（确保虚拟环境已激活）中，运行以下命令：

```bash
python train.py
```

这个脚本会执行所有复杂的数据处理和计算，包括数据清洗、价值分箱和图谱构建。运行成功后，它会生成（或覆盖）`data.pkl` 和 `graph.gpickle` 这两个模型文件。

**步骤 3：启动Web服务 (Start the Web Service)**

模型训练完成后，在同一个终端中运行以下命令来启动Web应用：

```bash
python app.py
```

如果一切顺利，您会看到类似以下的输出，表示Web服务器正在运行：
```
 * Running on http://127.0.0.1:5001
```

**步骤 4：查看前端 (View the Frontend)**

1.  打开您的网页浏览器（如Chrome, Firefox, Edge等）。
2.  在地址栏输入 `http://127.0.0.1:5001` 并回车。
3.  您现在应该能看到“融资租赁智能推荐系统”的前端界面。
4.  在页面的下拉菜单中选择您感兴趣的“省份”和“行业”。
5.  点击“获取推荐”按钮，页面下方将展示四个维度的推荐报告。

---

## 3. 文件说明 (File Descriptions)

*   **`train.py`**: 核心训练脚本。负责从CSV加载数据，进行所有预处理，并最终生成可供Web应用使用的模型文件。
*   **`app.py`**: 轻量级Web服务器。负责加载训练好的模型，并向前端提供API接口。
*   **`data_processing.py`**: 包含数据加载和清洗的函数。
*   **`feature_engineering.py`**: 包含特征工程（如价值分箱）的函数。
*   **`graph_builder.py`**: 包含构建主图谱和提取子图谱的函数。
*   **`recommender.py`**: 包含核心的PageRank推荐算法函数。
*   **`templates/index.html`**: 应用的前端页面文件。
*   **`data.pkl` / `graph.gpickle`**: 训练过程产生的二进制产物，包含了处理好的数据和图谱，供`app.py`快速加载。
