"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/13 下午1:02
@Email: 2909981736@qq.com
"""
import numpy as np

dataset = np.load('../data/reconstruct.npy')
ut = np.load('../data/ut.npy')

print(dataset.shape)
print(ut.shape)

