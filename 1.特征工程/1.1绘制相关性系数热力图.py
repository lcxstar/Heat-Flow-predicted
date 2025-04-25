# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 设置全局字体为 Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})
# 设置PDF文字以可编辑格式嵌入（Type 42字体）
plt.rcParams['pdf.fonttype'] = 42  # 关键参数，使文字不被转换为矢量图形
plt.rcParams['ps.fonttype'] = 42   # 同时设置PostScript兼容性（可选）

# 1. 加载并清洗数据
data = pd.read_excel('此处输入数据的文件名称.xlsx', sheet_name='All Table')
data = data.drop('编号', axis=1, errors='ignore')  #去除除数据列之外的所有干扰列
data = data.select_dtypes(include=['number'])

# 数据清洗：处理inf和NaN
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(axis=1, how='all')  # 删除全为NaN的列
data = data.dropna(axis=0, how='all')  # 删除全为NaN的行
data = data.fillna(data.mean())         # 填充剩余NaN

if 'Heat Flow' not in data.columns:
    raise ValueError("数据中缺少目标列 'Heat Flow'！")

# 2. 计算相关系数矩阵（带过滤）
def calculate_pvalues(df):
    n = df.shape[1]
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                col_i = df.iloc[:, i]
                col_j = df.iloc[:, j]
                # 跳过方差为0的列
                if col_i.nunique() == 1 or col_j.nunique() == 1:
                    p_matrix[i, j] = np.nan
                else:
                    try:
                        _, p = pearsonr(col_i, col_j)
                        p_matrix[i, j] = p
                    except:
                        p_matrix[i, j] = np.nan
    return pd.DataFrame(p_matrix, columns=df.columns, index=df.columns)

feature_corr = data.corr()
p_matrix = calculate_pvalues(data)

# 3. 生成标注矩阵
annot_matrix = np.empty_like(feature_corr, dtype='object')
for i in range(feature_corr.shape[0]):
    for j in range(feature_corr.shape[1]):
        r = feature_corr.iloc[i, j]
        p = p_matrix.iloc[i, j]
        if i == j:
            annot_matrix[i, j] = "——"
        else:
            if np.isnan(r) or np.isnan(p):
                annot_matrix[i, j] = "N/A"
            else:
                r_str = f"{r:.2f}\n"
                if p < 0.05:
                    p_str = "*"
                if p < 0.01:
                    p_str = "**"
                if p < 0.001:
                    p_str = "***"
                annot_matrix[i, j] = f"{r_str}{p_str}"

# 4. 绘制热力图
plt.figure(figsize=(7, 6))
mask = np.triu(np.ones_like(feature_corr, dtype=bool))


# 生成热力图并获取axes对象
ax = sns.heatmap(
    feature_corr,
    annot=annot_matrix,
    fmt='',
    cmap='coolwarm',
    linewidths=0.05,
    vmin=-1.0,
    vmax=1.0,
    cbar_kws={
        'label': 'Correlation Coefficient',
        'aspect': 20,            # 控制色标长宽比例
        'shrink': 0.9,           # 压缩色标高度
        'pad': 0.02,             # 色标与热力图的间距
        'ticks': np.arange(-1, 1.1, 0.4),  # 自定义刻度位置
        'drawedges': False       # 隐藏色标边缘线
    },
    annot_kws={'size': 6, 'color': 'black'}
)

# 获取色标对象并设置字体
cbar = ax.collections[0].colorbar

# 设置色标刻度标签字体大小
cbar.ax.tick_params(labelsize=6)

# 设置色标标题字体
cbar.set_label('Correlation Coefficient', 
               fontdict={'size':6, 'style': 'italic'},  # 字体大小和样式
               labelpad= 4)  # 调整标题位置（负值向上移动）


# 关闭所有刻度线显示
ax.tick_params(
    axis='both',         # 同时应用x/y轴
    which='both',       # 同时修改major和minor刻度
    length=0            # 设置刻度线长度为0
)
# plt.title('Feature Correlation Matrix with P-values', fontsize=6, pad=20)
# 自定义换行函数：在第一个空格或下划线处换行
def split_label(label, max_length=10):
    if len(label) <= max_length:
        return label
    # 查找第一个分隔符（空格或下划线）
    split_chars = [' ', '_']
    for char in split_chars:
        idx = label.find(char)
        if idx != -1:
            return f"{label[:idx]}\n{label[idx+1:]}"
    # 无分隔符时强制在中间换行
    mid = len(label) // 2
    return f"{label[:mid]}\n{label[mid:]}"

# 获取原始标签并处理换行
# x_labels = [split_label(label.get_text()) for label in ax.get_xticklabels()]
# y_labels = [split_label(label.get_text()) for label in ax.get_yticklabels()]

# # 设置X轴标签
# ax.set_xticks(np.arange(len(x_labels)))
# ax.set_xticklabels(
#     x_labels,
#     rotation=90,
#     ha='left',    # 左对齐
#     va='top',     # 顶部垂直对齐
#     fontsize=7,
#     linespacing=0.8  # 行间距
# )

# # 设置Y轴标签
# ax.set_yticks(np.arange(len(y_labels)))
# ax.set_yticklabels(
#     y_labels,
#     rotation=0,
#     ha='right',   # 右对齐（左侧留空）
#     va='center',  # 垂直居中
#     fontsize=7,
#     linespacing=0.8
# )

# # 调整布局防止标签被截断
# plt.subplots_adjust(left=0.25, bottom=0.25)  # 根据标签长度调整边距

plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)



plt.tight_layout()
# 保存图片，图片的清晰度设置
plt.savefig('Correlation Coefficient0329.png', dpi=300, bbox_inches='tight')
# 保存成pdf文件，用于矢量图编辑
plt.savefig('output.pdf', 
            format='pdf', 
            bbox_inches='tight',  # 自动裁剪空白区域
            transparent=True,     # 透明背景（按需启用）
            dpi=300)
plt.show()
