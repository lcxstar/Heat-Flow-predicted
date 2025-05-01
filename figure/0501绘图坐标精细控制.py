# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:07:12 2025

@author: liche
"""

# 实际-预测散点图（带刻度定制）这里的3.15绘制出来的图片底图宽度是8cm，实际图片宽度略小于8cm
plt.figure(figsize=(3.15, 3.15*(6/8))) 

# 绘制主体内容
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5)  # 虚线粗细调整为1.5，这里绘制的是一条斜率为1的参考线

# 坐标轴标签设置
plt.xlabel('Measured Heat Flow (mW/m²)', fontsize=7, labelpad=2)  # labelpad调整标签与轴的距离
plt.ylabel('Predicted Heat Flow (mW/m²)', fontsize=7, labelpad=2)

# 绘制图片内网格线
plt.grid(True, alpha=0.3) # 或者可以将True替换为axis='x'

# --- 刻度精细控制 ---
ax = plt.gca()  # 获取当前坐标轴

# 字号与刻度方向
ax.tick_params(axis='both', 
               which='major', 
               labelsize=6.5,         # 刻度标签字号
               length=2.5,            # 刻度线长度
               width=0.5,             # 刻度线粗细
               direction='in',        # 刻度朝向（'in'为朝内）
               pad=2,                 # 刻度标签与刻度线的距离
               bottom=True, top=True, # 显示上下刻度
               right=True, left=True) # 显示左右刻度

# 设置主刻度密度
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(5))  # X轴显示5个主刻度
ax.yaxis.set_major_locator(MaxNLocator(5))  # Y轴显示5个主刻度

# 设置坐标轴范围（按数据分布动态调整）
buffer = 0.1*(y.max()-y.min())  # 留10%的边距
plt.xlim(y.min()-buffer, y.max()+buffer)
plt.ylim(y.min()-buffer, y.max()+buffer)

# --- 其他格式调整 ---
plt.tight_layout(pad=1.5)  # pad参数增加图形与外框的间距

# 保存输出
plt.savefig('linear_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.savefig('linear_actual_vs_predicted.pdf', 
            format='pdf', 
            bbox_inches='tight',
            transparent=False,  # 地质图件通常需要白色背景
            dpi=300)
plt.close()
