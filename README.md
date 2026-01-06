# pandas-utils
高效的Pandas数据处理工具集，专注于**数据清洗、自动化EDA、大模型微调数据集预处理**，适配工业级数据处理场景与面试核心考点。

## 核心功能
### 1. 数据清洗（DataCleaner）
- 智能缺失值处理（自动选择策略：高缺失率删列、数值列中位数填充、类别列众数填充）
- 重复值检测与删除
- 异常值处理（IQR过滤/截断）
- 批量数据类型转换

### 2. 自动化EDA（EDAAnalyzer）
- 一键生成基础统计报告（数据形状、缺失值、数据类型分布等）
- 数值列分布可视化
- 相关性热力图生成

### 3. 数据格式转换（DataConverter）
- CSV/JSONL格式互转（适配大模型微调数据集规范）
- 一键转换为大模型微调标准格式 `{"instruction": "", "input": "", "output": ""}`

## 安装与使用
### 安装依赖
```bash
pip install -r requirements.txt

import pandas as pd
from pandas_utils import DataCleaner, EDAAnalyzer, DataConverter

# 加载数据
df = pd.read_csv("your_data.csv")

# 数据清洗
cleaner = DataCleaner(df)
df_clean = cleaner.handle_missing_values(strategy="auto")
df_clean = cleaner.handle_duplicates()

# 自动化EDA
analyzer = EDAAnalyzer(df_clean)
analyzer.basic_stats()
analyzer.plot_numeric_dist(save_path="./numeric_dist.png")

# 转换为大模型微调格式
DataConverter.format_for_llm_finetune(
    input_path="./cleaned_data.csv",
    output_path="./finetune_data.jsonl",
    mapping={"question": "instruction", "answer": "output"}
)
