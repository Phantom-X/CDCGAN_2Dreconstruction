"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/13 上午9:12
@Email: 2909981736@qq.com
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/data_notabnormal.csv")

print(df.head())

reconstruct_data = []
conditional_data = []
for col in df.columns:
    reconstruct_data.append(df[col].values.reshape(64, 64).tolist())
    conditional_data.append(list(map(float, col.split('@'))))
print(reconstruct_data)
print(conditional_data)
reconstruct_data_npy = np.array(reconstruct_data)
conditional_data_npy = np.array(conditional_data)
np.save("../data/reconstruct.npy", reconstruct_data_npy)
np.save("../data/ut.npy", conditional_data_npy)
