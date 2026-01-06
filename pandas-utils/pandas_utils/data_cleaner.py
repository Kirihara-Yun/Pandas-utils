import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union


class DataCleaner:
    """
    数据清洗工具类
    提供缺失值处理、重复值处理、异常值检测与处理、数据类型转换等核心功能
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clean_history = []  # 记录清洗操作历史

    def handle_missing_values(
        self,
        strategy: str = "auto",
        fill_values: Optional[Dict[str, Union[int, float, str]]] = None,
        drop_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        处理缺失值
        :param strategy: 处理策略 - auto: 自动选择（高缺失率删列，数值列中位数，类别列众数）
                           fill: 自定义填充值
                           drop: 删除缺失值
        :param fill_values: 自定义填充字典 {列名: 填充值}
        :param drop_threshold: 列缺失率阈值，超过则删除列
        :return: 清洗后的DataFrame
        """
        # 删除高缺失率列
        missing_ratio = self.df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.clean_history.append(f"删除高缺失率列: {cols_to_drop}")

        if strategy == "drop":
            self.df = self.df.dropna()
            self.clean_history.append("删除所有含缺失值的行")
        elif strategy == "fill" and fill_values:
            for col, val in fill_values.items():
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(val)
            self.clean_history.append(f"自定义填充缺失值: {fill_values}")
        elif strategy == "auto":
            # 数值列用中位数填充，类别列用众数填充
            for col in self.df.columns:
                if self.df[col].dtype in ["int64", "float64"]:
                    fill_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_val)
                else:
                    fill_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(fill_val)
            self.clean_history.append("自动填充缺失值（数值列中位数，类别列众数）")
        else:
            raise ValueError("无效的策略，请选择 auto/fill/drop")
        
        return self.df

    def handle_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理重复值
        :param subset: 指定去重列，None则按所有列去重
        :return: 清洗后的DataFrame
        """
        duplicate_count = self.df.duplicated(subset=subset).sum()
        if duplicate_count > 0:
            self.df = self.df.drop_duplicates(subset=subset)
            self.clean_history.append(f"删除重复值 {duplicate_count} 行")
        return self.df

    def handle_outliers(self, cols: List[str], method: str = "iqr") -> pd.DataFrame:
        """
        处理数值列异常值
        :param cols: 需要处理的数值列列表
        :param method: iqr: 基于四分位数范围过滤 | clip: 截断异常值
        :return: 清洗后的DataFrame
        """
        for col in cols:
            if col not in self.df.columns or self.df[col].dtype not in ["int64", "float64"]:
                continue
            
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if method == "iqr":
                before_count = len(self.df)
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                after_count = len(self.df)
                self.clean_history.append(f"{col}列基于IQR过滤异常值，删除 {before_count - after_count} 行")
            elif method == "clip":
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                self.clean_history.append(f"{col}列截断异常值至[{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return self.df

    def convert_dtypes(self, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        批量转换数据类型
        :param dtype_mapping: {列名: 目标类型} 支持 int/float/str/category
        :return: 清洗后的DataFrame
        """
        for col, dtype in dtype_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == "category":
                        self.df[col] = self.df[col].astype("category")
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                    self.clean_history.append(f"{col}列转换为{dtype}类型")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"列{col}转换{dtype}失败: {str(e)}")
        return self.df

    def get_clean_history(self) -> List[str]:
        """返回清洗操作历史"""
        return self.clean_history