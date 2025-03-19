# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:41:02 2025
生成一个等值线图，生成一个散点图
@author: liche
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 0. 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 1. 设置文件路径（修改部分）
csv_path = r"等值线\数据文件\路径.csv"
csv_path2 = r"散点\数据文件\路径.csv"  # 需要用户修改路径
output_dir = os.path.dirname(csv_path)  # 获取CSV所在目录

# 2. 读取数据（自动获取列名）
df = pd.read_csv(csv_path)
column_name = df.columns[2]  # 第三列名称
base_name = os.path.splitext(os.path.basename(csv_path))[0]

lons = df[df.columns[0]].values  # 自动获取经度列
lats = df[df.columns[1]].values  # 自动获取纬度列
z = df[column_name].values

# 读取散点数据
df2 = pd.read_csv(csv_path2)
lons2 = df2.iloc[:, 0].values  # 第一列经度
lats2 = df2.iloc[:, 1].values  # 第二列纬度
z2 = df2.iloc[:, 2].values     # 第三列热流值

# 3. 生成规则网格（扩大范围10%）
buffer = 0.2
lon_min, lon_max = lons.min()-buffer, lons.max()+buffer
lat_min, lat_max = lats.min()-buffer, lats.max()+buffer

xi = np.arange(lon_min, lon_max, 0.01)
yi = np.arange(lat_min, lat_max, 0.01)
xi, yi = np.meshgrid(xi, yi)

# 4. 三次样条插值
zi = griddata((lons, lats), z, (xi, yi), method='cubic')

# 5. 读取边界数据（路径需要确认）
border_path = os.path.join(output_dir, "盆地边界坐标_经纬度格式.csv")  # 假设边界文件在同目录
df_border = pd.read_csv(border_path)
border_points = list(zip(df_border['lon'], df_border['lat']))
polygon = Polygon(border_points + [border_points[0]])

# 筛选散点数据（仅保留边界内）
points2 = [Point(lon, lat) for lon, lat in zip(lons2, lats2)]
mask2 = np.array([polygon.contains(point) for point in points2])
lons2_filtered = lons2[mask2]
lats2_filtered = lats2[mask2]
z2_filtered = z2[mask2]

# 6. 裁剪网格数据
points = [Point(x, y) for x, y in zip(xi.ravel(), yi.ravel())]
gdf_grid = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
mask = gdf_grid.within(polygon)
zi_masked = zi.ravel().copy()
zi_masked[~mask] = np.nan
zi_clipped = zi_masked.reshape(zi.shape)

# 7. 绘图设置
plt.figure(figsize=(20, 10))
plt.title(column_name, fontsize=20, pad=15)
contour_fill = plt.contourf(xi, yi, zi_clipped, levels=20, cmap='jet')

# 8. 添加等值线标注
contour_lines = plt.contour(xi, yi, zi_clipped, levels=20, colors='k', linewidths=0.5)
selected_levels = contour_lines.levels[::2]
plt.clabel(contour_lines, levels=selected_levels, inline=True, 
          fontsize=12, fmt='%d', colors='black')

# 9. 叠加散点数据
plt.scatter(lons2_filtered, lats2_filtered, c=z2_filtered, 
            cmap=contour_fill.cmap, norm=contour_fill.norm,
            edgecolors='k', linewidths=0.5, s=40, zorder=3)

# 10. 添加边界和坐标轴
plt.plot(*polygon.exterior.xy, color='black', linewidth=1.5, 
         label='Basin Boundary', zorder=2)
plt.legend(fontsize=20)
ax = plt.gca()
ax.set_xlim(82, 92)
ax.set_ylim(43, 47)
ax.set_xlabel('Longitude (°E)', fontsize=20, labelpad=8)
ax.set_ylabel('Latitude (°N)', fontsize=20, labelpad=8)

# 11. 色标设置
axins = inset_axes(ax, width="40%", height="4%", loc='lower left', borderpad=4)
cbar = plt.colorbar(contour_fill, cax=axins, orientation='horizontal', aspect=20)
cbar.set_label(f'{column_name} (mW/$^2$)', fontsize=14, labelpad=5)
cbar.ax.tick_params(labelsize=14, direction='in')
axins.set_position([0.65, 0.82, 0.3, 0.03])

# 12. 保存到原始CSV目录
output_path = os.path.join(output_dir, f"{base_name}.pdf")
plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
plt.close()

print(f"文件已保存至：{output_path}")
