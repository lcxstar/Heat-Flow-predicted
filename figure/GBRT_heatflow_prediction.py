# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:30:48 2025
@author: liche
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor  # 修改1：导入GBRT
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# 3. 贝叶斯优化框架（修改为GBRT参数）
optimization_history = pd.DataFrame(columns=[
    'n_estimators', 'learning_rate', 'max_depth', 
    'min_samples_split', 'min_samples_leaf', 'r2'
])

def gbr_bayesian_eval(n_estimators, learning_rate, max_depth, 
                     min_samples_split, min_samples_leaf):
    """GBRT贝叶斯优化评估函数"""
    # 参数处理（将浮点数转换为整数）
    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': max(learning_rate, 1e-3),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
    }
    
    # 交叉验证评估
    model = GradientBoostingRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                               scoring='r2', n_jobs=-1)
    r2 = np.mean(cv_scores)
    
    # 记录优化过程
    optimization_history.loc[len(optimization_history)] = {
        'n_estimators': params['n_estimators'],
        'learning_rate': params['learning_rate'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'r2': r2
    }
    
    return r2

# 4. 执行贝叶斯优化（GBRT参数空间）
optimizer = BayesianOptimization(
    f=gbr_bayesian_eval,
    pbounds={
        'n_estimators': (500,1500),
        'learning_rate': (0.005, 0.1),
        'max_depth': (5, 15),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (2, 10)
    },
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=20, n_iter=100)

# 5. 获取并训练最终模型
best_params = {
    'n_estimators': int(optimizer.max['params']['n_estimators']),
    'learning_rate': optimizer.max['params']['learning_rate'],
    'max_depth': int(optimizer.max['params']['max_depth']),
    'min_samples_split': int(optimizer.max['params']['min_samples_split']),
    'min_samples_leaf': int(optimizer.max['params']['min_samples_leaf'])
}

final_model = GradientBoostingRegressor(**best_params)
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

# # 7. 可视化（调整标题和文件名）
# plt.figure(figsize=(12, 6))
# plt.plot(optimization_history.index, optimization_history['r2'], 
#          marker='o', linestyle='--', color='#2c7bb6')
# plt.title('GBRT Optimization Progress', fontsize=14)
# plt.xlabel('Iteration', fontsize=12)
# plt.ylabel('Cross-Val R2 Score', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.savefig('gbrt_optimization_progress.png', dpi=300, bbox_inches='tight')
# plt.close()

# 8. 保存结果（调整文件名）
pd.DataFrame([best_params]).to_csv('gbrt_best_params.csv', index=False)
pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv('gbrt_metrics.csv', index=False)
optimization_history.to_csv('gbrt_optimization_history.csv', index=False)
joblib.dump(final_model, 'optimized_gbrt_model.pkl')

print("GBRT优化完成！生成文件列表：")
print("- gbrt_best_params.csv        # 最佳参数组合")
print("- gbrt_metrics.csv            # 评估指标")
print("- gbrt_optimization_history.csv # 完整优化记录")
print("- optimized_gbrt_model.pkl     # 训练好的GBRT模型")
print("- gbrt_optimization_progress.png # 优化进度图")
