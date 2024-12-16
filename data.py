import pandas as pd

from config import DATA_PATH, SAMPLE_PATH

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
