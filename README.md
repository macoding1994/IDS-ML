# IDS-ML



**定义目标**： 确定要解决的问题是分类、回归、聚类还是其他任务

- **分类**：判断邮件是否是垃圾邮件
- **回归**：预测房价
- **聚类**：用户分群



**准备数据**：

- 收集数据集（如 CSV 文件）
- 清洗数据：处理缺失值、异常值
- 特征工程：对数据进行编码、标准化或归一化
- 划分数据集：通常分为训练集和测试集（如 80%:20%）



根据任务选择适合的机器学习算法：

- **分类**：
  - 逻辑回归（Logistic Regression）
  - 决策树（Decision Tree）
  - 随机森林（Random Forest）
  - XGBoost、LightGBM
  - 神经网络（Neural Network）
- **回归**：
  - 线性回归（Linear Regression）
  - 支持向量机回归（SVR）
  - 梯度提升回归（XGBoost, LightGBM）
- **聚类**：
  - K-Means
  - DBSCAN
  - 层次聚类



https://www.unb.ca/cic/datasets/ids-2017.html

**Day, Date, Description, Size (GB)**

- Monday, Normal Activity, 11.0G
- Tuesday, attacks + Normal Activity, 11G
- Wednesday, attacks + Normal Activity, 13G
- Thursday, attacks + Normal Activity, 7.8G
- Friday, attacks + Normal Activity, 8.3G





