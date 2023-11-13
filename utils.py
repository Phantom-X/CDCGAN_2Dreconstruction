# coding: utf-8
import os

path = os.path

OUTPUT_DIR = 'output'
FAKE_MNIST_DIR = os.path.join(OUTPUT_DIR, 'fake-reconstruct_test')
REAL_MNIST_DIR = os.path.join(OUTPUT_DIR, 'real-reconstruct_test')

_paths = [OUTPUT_DIR, FAKE_MNIST_DIR, REAL_MNIST_DIR]

for p in _paths:
    if not os.path.exists(p):
        os.mkdir(p)
