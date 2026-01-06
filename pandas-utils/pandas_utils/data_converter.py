import pandas as pd
import json
from typing import Optional, Dict, List
import os


class DataConverter:
    """
    数据格式转换工具
    专注于CSV/JSON/JSONL格式互转，适配大模型微调数据集规范
    """

    @staticmethod
    def csv_to_jsonl(
        input_path: str,
        output_path: str,
        orient: str = "records",
        encoding: str = "utf-8",
        columns: Optional[List[str]] = None
    ):
        """
        CSV转JSONL（大模型微调常用格式）
        :param input_path: CSV文件路径
        :param output_path: JSONL输出路径
        :param orient: 输出格式，records为每行一个JSON对象
        :param encoding: 文件编码
        :param columns: 指定转换列，None则转换所有列
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        df = pd.read_csv(input_path, encoding=encoding)
        if columns:
            df = df[columns]
        
        # 写入JSONL
        with open(output_path, "w", encoding=encoding) as f:
            for record in df.to_dict(orient=orient):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def jsonl_to_csv(
        input_path: str,
        output_path: str,
        encoding: str = "utf-8"
    ):
        """JSONL转CSV"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        records = []
        with open(input_path, "r", encoding=encoding) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding=encoding)

    @staticmethod
    def format_for_llm_finetune(
        input_path: str,
        output_path: str,
        mapping: Dict[str, str],
        encoding: str = "utf-8"
    ):
        """
        转换为大模型微调标准格式 {"instruction": "", "input": "", "output": ""}
        :param input_path: 输入JSONL/CSV文件路径
        :param output_path: 输出JSONL路径
        :param mapping: 字段映射 {原字段名: 目标字段名} 如 {"问题": "instruction", "回答": "output"}
        """
        # 识别文件类型
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(input_path, encoding=encoding)
        elif file_ext == ".jsonl":
            records = []
            with open(input_path, "r", encoding=encoding) as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            df = pd.DataFrame(records)
        else:
            raise ValueError("仅支持CSV/JSONL格式输入")
        
        # 字段映射
        required_fields = ["instruction", "input", "output"]
        for target in required_fields:
            if target not in mapping.values():
                if target == "input":  # input可选，默认空字符串
                    df[target] = ""
                else:
                    raise ValueError(f"映射缺少必填字段: {target}")
        
        df = df.rename(columns=mapping)[required_fields]
        
        # 写入JSONL
        with open(output_path, "w", encoding=encoding) as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")