import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


# import xgboost as xgb
# from xgboost import plot_importance


def merge_csv_files(path="data"):
    # 获取所有 CSV 文件的路径（假设它们在当前目录下）
    csv_files = glob.glob(f"{path}/*.csv")  # 替换为你的目录路径
    # 初始化一个空列表来存储数据框
    dataframes = []
    # 读取每个文件并追加到列表中
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    # 将所有数据框合并为一个
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.columns = combined_df.columns.str.strip()
    # 保存为新的 CSV 文件
    combined_df.to_csv("data/CICIDS2017.csv", index=False)


# 文件路径
DATA_PATH = './data/CICIDS2017.csv'
SAMPLE_PATH = './data/CICIDS2017_sample.csv'

# 数据读取与采样
df = pd.read_csv(DATA_PATH)
# DATA_PATH中数据如下：
# BENIGN                        2273097
# DoS Hulk                       231073
# PortScan                       158930
# DDoS                           128027
# DoS GoldenEye                   10293
# FTP-Patator                      7938
# SSH-Patator                      5897
# DoS slowloris                    5796
# DoS Slowhttptest                 5499
# Bot                              1966
# Web Attack � Brute Force         1507
# Web Attack � XSS                  652
# Infiltration                       36
# Web Attack � Sql Injection         21
# Heartbleed                         11


# Step 1: 随机下采样多数类，保留少数类完整数据
def downsample_data(df, target_col, majority_classes, minority_classes, sampling_ratios):
    # 保存少数类
    df_minority = df[df[target_col].isin(minority_classes)]
    # 对多数类进行下采样
    dfs = []
    for label, frac in sampling_ratios.items():
        df_majority = df[df[target_col] == label]
        dfs.append(df_majority.sample(frac=frac, random_state=42))  # 随机下采样
    # 合并多数类和少数类数据
    df_balanced = pd.concat(dfs + [df_minority])
    return df_balanced


# 定义类别
majority_classes = ["BENIGN", "DoS Hulk", "PortScan", "DDoS"]
minority_classes = ["DoS GoldenEye", "FTP-Patator", "SSH-Patator", "DoS slowloris",
                    "DoS Slowhttptest", "Bot", "Web Attack � Brute Force",
                    "Web Attack � XSS", "Infiltration", "Web Attack � Sql Injection",
                    "Heartbleed"]

# 定义下采样比例
sampling_ratios = {
    "BENIGN": 0.01,  # 保留 1%
    "DoS Hulk": 0.05,  # 保留 5%
    "PortScan": 0.05,  # 保留 5%
    "DDoS": 0.05  # 保留 5%
}

# 执行下采样
df_balanced = downsample_data(df, "Label", majority_classes, minority_classes, sampling_ratios)
df_balanced.to_csv(SAMPLE_PATH)
# 采样后的数据
# BENIGN                        22731
# DoS Hulk                      11554
# DoS GoldenEye                 10293
# PortScan                       7946
# FTP-Patator                    7938
# DDoS                           6401
# SSH-Patator                    5897
# DoS slowloris                  5796
# DoS Slowhttptest               5499
# Bot                            1966
# Web Attack � Brute Force       1507
# Web Attack � XSS                652
# Infiltration                     36
# Web Attack � Sql Injection       21
# Heartbleed                       11
# Name: Label, dtype: int64


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

# # 检查和处理 NaN 和 Inf 值
# X_train = X_train.fillna(0)
# X_test = X_test.fillna(0)
#
# X_train = X_train.replace([np.inf, -np.inf], 0)
# X_test = X_test.replace([np.inf, -np.inf], 0)
#
# # 确保所有特征为数值类型
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)
#
# # 检查数据是否正常
# print("NaN in X_train:", np.isnan(X_train).any())
# print("NaN in X_test:", np.isnan(X_test).any())
# print("Inf in X_train:", np.isinf(X_train).any())
# print("Inf in X_test:", np.isinf(X_test).any())
#
# # 验证数据统计信息
# print(X_train.describe())
# print(X_test.describe())
#
# # 检查 y_train
# print("y_train type:", type(y_train))
# print("y_train dtype:", y_train.dtype)
# print("y_train unique values:", pd.Series(y_train).unique())
#
# # 转换为整数（如果需要）
# if y_train.dtype == 'float64' or y_train.dtype == 'object':
#     y_train = y_train.astype(int)
#     y_test = y_test.astype(int)
#
# # 检查和处理无效值
# if pd.Series(y_train).isnull().any():
#     print("NaN values found in y_train, filling with forward fill.")
#     y_train = pd.Series(y_train).fillna(method='ffill').astype(int)
#
# # 再次检查修复结果
# print("y_train unique values after processing:", pd.Series(y_train).unique())

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
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
