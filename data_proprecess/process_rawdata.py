"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/10 下午4:26
@Email: 2909981736@qq.com
"""
import pandas as pd
import numpy as np
import csv
from io import StringIO


# rawdata = pd.read_csv("../data/rawdata_backup.csv", skiprows=8, usecols=range(5822))
# rawdata.to_csv('../data/rawdata.csv', index=False)


def colnamestrans():
    coldata = pd.read_csv("../data/rawdata_backup.csv", skiprows=8, nrows=1, header=None)
    colnames = coldata.iloc[0].values.tolist()
    # print(coldata.iloc[0].values.tolist())
    new_colnames = []
    new_colnames.extend(['% x', 'y'])
    with open("../data/rawdata.csv", 'r') as f:
        lines = f.readlines()
        flag = 1
        for c in colnames[2:]:
            if flag == 1:
                t = c.split("=")[1]
                flag += 1
            else:
                u = c.split("=")[1]
                ut = u + "@" + t
                new_colnames.append(ut)
                flag = 1
        print(new_colnames)
        print(len(new_colnames))
        lines[0] = ','.join(new_colnames)+'\n'
        print(lines[0].find('\n'))
    with open("../data/rawdata.csv", 'w') as f:
        f.writelines(lines)


# # ut
# colnamestrans()
#
rawdata = pd.read_csv('../data/rawdata.csv')
print(rawdata)
print(len(rawdata.columns.tolist()))
print(rawdata.isnull().sum().any())
rawdata.fillna(0, inplace=True)
print(rawdata.isnull().sum().any())
rawdata.to_csv("../data/data_notNan.csv", index=False)

