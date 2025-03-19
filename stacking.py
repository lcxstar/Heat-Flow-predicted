# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:30:48 2025
@author: liche
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor  # 修改为KNN回归模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV

# 1. 数据加载与预处理（保持不变）
data = pd.read_excel('0317特征参数16个.xlsx', sheet_name='All Table')
data = data.drop('编号', axis=1, errors='ignore')
data = data.select_dtypes(include=['number'])
data = data.dropna(subset=['Heat Flow'])
data = data[data["Heat Flow"] < 60]
X = data.drop('Heat Flow', axis=1)
y = data['Heat Flow']

# 2. 数据标准化（树模型可不做标准化，但为保持流程统一保留）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 贝叶斯优化框架（修改为KNN参数）
optimization_history = pd.DataFrame(columns=[
    'n_neighbors', 'p', 'weights', 'r2'  # KNN核心参数
])

def knn_bayesian_eval(n_neighbors, p, weights):
    """KNN贝叶斯优化评估函数"""
    # 参数处理（将浮点数转换为整数）
    params = {
        'n_neighbors': int(n_neighbors),
        'p': int(p),
        'weights': 'uniform' if weights < 0.5 else 'distance'
    }
    
    # 交叉验证评估
    model = KNeighborsRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                               scoring='r2', n_jobs=-1)
    r2 = np.mean(cv_scores)
    
    # 记录优化过程
    optimization_history.loc[len(optimization_history)] = {
        'n_neighbors': params['n_neighbors'],
        'p': params['p'],
        'weights': params['weights'],
        'r2': r2
    }
    return r2

# 4. 执行贝叶斯优化（KNN参数空间）
optimizer = BayesianOptimization(
    f=knn_bayesian_eval,
    pbounds={
        'n_neighbors': (3, 20),    # 邻居数量范围
        'p': (1, 2),               # 距离度量（1:曼哈顿，2:欧氏）
        'weights': (0, 1)          # 权重类型（0:uniform，1:distance）
    },
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=15, n_iter=100)

# 5. 获取并训练最终模型
best_params = {
    'n_neighbors': int(optimizer.max['params']['n_neighbors']),
    'p': int(optimizer.max['params']['p']),
    'weights': 'uniform' if optimizer.max['params']['weights'] < 0.5 else 'distance'
}
bayesian_best = {
    'n_neighbors': int(optimizer.max['params']['n_neighbors']),
    'p': int(optimizer.max['params']['p']),
    'weights': 'uniform' if optimizer.max['params']['weights'] < 0.5 else 'distance'
}

# 设置动态参数网格（基于贝叶斯优化结果）
# 修改后的网格搜索参数设置
param_grid = {
    'n_neighbors': [max(3, bayesian_best['n_neighbors']-2),
                    bayesian_best['n_neighbors'],
                    min(50, bayesian_best['n_neighbors']+2)],
    'p': [max(1, bayesian_best['p']-1),  # 确保p不小于1
          bayesian_best['p'],
          bayesian_best['p']+1],
    'weights': [bayesian_best['weights']]
}

# 执行网格搜索
grid_search = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("贝叶斯优化最佳参数:", bayesian_best)
print("网格搜索最佳参数:", grid_search.best_params_)
final_model = KNeighborsRegressor(**best_params)
final_model.fit(X_train, y_train)

# 6. 模型评估与结果保存
y_pred = final_model.predict(X_test)
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R2': r2_score(y_test, y_pred)
}
print(best_params, metrics)

# 7. 可视化（调整标题和文件名）
# plt.figure(figsize=(12, 6))
# plt.plot(optimization_history.index, optimization_history['r2'], 
#          marker='o', linestyle='--', color='#2c7bb6')
# plt.title('KNN Optimization Progress', fontsize=14)
# plt.xlabel('Iteration', fontsize=12)
# plt.ylabel('Cross-Val R2 Score', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.savefig('knn_optimization_progress.png', dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(14, 8))
# sns.heatmap(
#     optimization_history[['n_neighbors', 'p', 'weights', 'r2']].corr(),
#     annot=True,
#     cmap='coolwarm',
#     center=0,
#     fmt=".2f",
#     annot_kws={'size': 10}
# )
# plt.title('KNN Parameter Correlation Heatmap', fontsize=14)
# plt.xticks(fontsize=10, rotation=45)
# plt.yticks(fontsize=10)
# plt.savefig('knn_param_correlation.png', dpi=300, bbox_inches='tight')
# plt.close()

# 8. 保存结果（调整文件名）
pd.DataFrame([best_params]).to_csv('knn_best_params.csv', index=False)
pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv('knn_metrics.csv', index=False)
optimization_history.to_csv('knn_optimization_history.csv', index=False)
joblib.dump(final_model, 'optimized_knn_model0.5536.pkl')

# print("KNN优化完成！生成文件列表：")
# print("- knn_best_params.csv        # 最佳参数组合")
# print("- knn_metrics.csv            # 评估指标")
# print("- knn_optimization_history.csv # 完整优化记录")
# print("- optimized_knn_model.pkl     # 训练好的KNN模型")
# print("- knn_optimization_progress.png # 优化进度图")
# print("- knn_param_correlation.png  # 参数相关性图")
