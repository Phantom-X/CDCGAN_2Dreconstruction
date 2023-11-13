"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/13 上午9:12
@Email: 2909981736@qq.com
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/data_notabnormal.csv")

print(df)

# reconstruct_data = []
# conditional_data = []
# for col in df.columns:
#     reconstruct_data.append(df[col].values.reshape(64, 64))
#     conditional_data.append(list(map(float, col.split('@'))))
#     break
# print(reconstruct_data)
# print(conditional_data)
