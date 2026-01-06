import pandas as pd
from pandas_utils import DataCleaner, EDAAnalyzer, DataConverter

# 加载示例数据（Titanic）
def load_sample_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

if __name__ == "__main__":
    # 1. 加载数据
    df = load_sample_data()
    print("原始数据形状:", df.shape)

    # 2. 数据清洗
    cleaner = DataCleaner(df)
    # 处理缺失值（自动策略）
    df_clean = cleaner.handle_missing_values(strategy="auto")
    # 处理重复值
    df_clean = cleaner.handle_duplicates()
    # 处理异常值（Fare列）
    df_clean = cleaner.handle_outliers(cols=["Fare"], method="clip")
    # 转换数据类型
    df_clean = cleaner.convert_dtypes({"Pclass": "category", "Survived": "int8"})
    print("清洗后数据形状:", df_clean.shape)
    print("清洗操作历史:", cleaner.get_clean_history())

    # 3. 自动化EDA
    analyzer = EDAAnalyzer(df_clean)
    # 基础统计
    basic_stats = analyzer.basic_stats()
    print("基础统计:", basic_stats["数据形状"])
    # 绘制数值列分布
    analyzer.plot_numeric_dist(save_path="./numeric_dist.png")
    # 绘制相关性热力图
    analyzer.plot_correlation(save_path="./correlation.png")
    print("EDA报告生成完成，包含:", list(analyzer.get_report().keys()))

    # 4. 格式转换（适配大模型微调）
    # 保存清洗后的数据为CSV
    df_clean.to_csv("./cleaned_titanic.csv", index=False)
    # CSV转JSONL
    DataConverter.csv_to_jsonl(
        input_path="./cleaned_titanic.csv",
        output_path="./titanic.jsonl",
        columns=["Pclass", "Sex", "Age", "Survived"]
    )
    # 转换为大模型微调格式
    DataConverter.format_for_llm_finetune(
        input_path="./titanic.jsonl",
        output_path="./titanic_finetune.jsonl",
        mapping={"Pclass": "instruction", "Age": "input", "Survived": "output"}
    )
    print("数据格式转换完成，生成 titanic_finetune.jsonl")