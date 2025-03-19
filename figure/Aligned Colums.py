# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 20:45:52 2025

@author: liche
"""

import pandas as pd

# 读取文件1和文件2
df1 = pd.read_excel("File 1.xlsx")  # 替换为实际路径
df2 = pd.read_excel("File 2.xlsx")  # 替换为实际路径

# 获取File 1的列顺序列表
target_columns = df1.columns.tolist()

# 检查File 2是否包含File 1的所有列（可选，根据需求调整）
missing_cols = [col for col in target_columns if col not in df2.columns]
if missing_cols:
    raise ValueError(f"File 2缺少以下列：{missing_cols}")

# 调整File 2的列顺序并删除多余列
df2_aligned = df2[target_columns]  # 按File 1顺序筛选并排序

# Save as File 3
df2_aligned.to_excel("File 3.xlsx", index=False)
