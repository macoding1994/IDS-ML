import glob

import pandas as pd


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



if __name__ == '__main__':
    merge_csv_files()