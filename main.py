import warnings

from config import DATA_PATH, SAMPLE_PATH

warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import plot_importance

# Step 2: 对数值特征进行标准化（假设已有其他数值特征）
# 如果你的数据有更多列，可以添加具体的标准化步骤
#  的目的是对数值特征进行 标准化 或 归一化。标准化是一种数据预处理方法，在机器学习任务中非常常见。
# 消除特征量纲的影响:标准化后，所有特征的值都被缩放到相同的范围（如 [0,1]），避免模型对某些特征产生偏倚
# 提高模型收敛速度:对于基于梯度下降的算法（如逻辑回归、神经网络等），标准化可以加快训练过程中的收敛速度
# 采用的是 归一化（(x - x.min()) / (x.max() - x.min())），因为归一化适用于多数情况下的数据预处理需求，尤其是特征范围差异较大时
# 标准化: 将不同规则的数据，通过同一标准汇聚起来
# 归一化：将数据取值范围统一按0-1处理
df_balanced = pd.read_csv(SAMPLE_PATH)
numeric_features = df_balanced.dtypes[df_balanced.dtypes != 'object'].index
df_balanced[numeric_features] = df_balanced[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
# Fill empty values by 0
df = df_balanced.fillna(0)
# # Step 3: 将类别标签转换为数值编码

labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
X = df.drop(['Label'], axis=1).values
y = df.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)
# Step 4: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# Decision Tree training and prediction
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
y_predict = dt.predict(X_test)
y_true = y_test
print('Accuracy of DT: ' + str(dt_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of DT: ' + (str(precision)))
print('Recall of DT: ' + (str(recall)))
print('F1-score of DT: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("Decision Tree training and prediction")
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# Random Forest training and prediction
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
y_predict = rf.predict(X_test)
y_true = y_test
print('Accuracy of RF: ' + str(rf_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of RF: ' + (str(precision)))
print('Recall of RF: ' + (str(recall)))
print('F1-score of RF: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("Random Forest training and prediction")
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# XGBoost training and prediction
xg = xgb.XGBClassifier(n_estimators=10)
xg.fit(X_train, y_train)
xg_score = xg.score(X_test, y_test)
y_predict = xg.predict(X_test)
y_true = y_test
print('Accuracy of XGBoost: ' + str(xg_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of XGBoost: ' + (str(precision)))
print('Recall of XGBoost: ' + (str(recall)))
print('F1-score of XGBoost: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("XGBoost training and prediction")
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# LogisticRegression training and prediction
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
y_predict = lr.predict(X_test)
y_true = y_test
print('Accuracy of lr: ' + str(lr_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of lr: ' + (str(precision)))
print('Recall of lr: ' + (str(recall)))
print('F1-score of lr: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("LogisticRegression training and prediction")
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
