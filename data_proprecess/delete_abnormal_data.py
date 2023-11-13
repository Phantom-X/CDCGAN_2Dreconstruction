"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/13 上午9:14
@Email: 2909981736@qq.com
"""
import pandas as pd
import numpy as np

notnan_data = pd.read_csv("../data/data_notNan.csv")

print(notnan_data.head())

delcol = ['% x', 'y']
for col in notnan_data.columns[2:]:
    u, t = map(float, col.split("@"))
    if t == 0:
        delcol.append(col)

res_df = notnan_data.drop(delcol, axis=1)

print(res_df)

res_df.to_csv('../data/data_notabnormal.csv', index=False)
