import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Union
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 适配多环境，避免中文乱码
plt.rcParams['axes.unicode_minus'] = False


class EDAAnalyzer:
    """
    自动化探索性数据分析工具
    提供基础统计、分布可视化、相关性分析等功能
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analysis_report = {}

    def basic_stats(self) -> Dict:
        """生成基础统计报告"""
        stats = {
            "数据形状": self.df.shape,
            "数值列统计": self.df.describe().to_dict(),
            "类别列统计": self.df.select_dtypes(include=["object", "category"]).describe().to_dict(),
            "缺失值统计": self.df.isnull().sum().to_dict(),
            "数据类型分布": self.df.dtypes.value_counts().to_dict()
        }
        self.analysis_report["基础统计"] = stats
        return stats

    def plot_numeric_dist(self, cols: Optional[List[str]] = None, save_path: str = "./numeric_dist.png"):
        """
        绘制数值列分布直方图
        :param cols: 指定列，None则选所有数值列
        :param save_path: 图片保存路径
        """
        if not cols:
            cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not cols:
            raise ValueError("无数值列可绘制")
        
        n_cols = 2
        n_rows = (len(cols) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(cols):
            self.df[col].hist(ax=axes[idx], bins=30, edgecolor="black")
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        
        # 隐藏多余子图
        for idx in range(len(cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        self.analysis_report["数值列分布图表"] = save_path

    def plot_correlation(self, save_path: str = "./correlation.png"):
        """绘制相关性热力图"""
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_df.shape[1] < 2:
            raise ValueError("数值列数量不足，无法绘制相关性热力图")
        
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm")
        
        # 添加刻度和标签
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.columns)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")
        
        ax.set_title("Correlation Matrix")
        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        self.analysis_report["相关性热力图"] = save_path

    def get_report(self) -> Dict:
        """返回完整分析报告"""
        return self.analysis_report