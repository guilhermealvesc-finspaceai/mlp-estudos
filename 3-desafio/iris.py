import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
indices = list(range(len(iris['data'])))
random.shuffle(indices)

iris['data'] = iris['data'][indices]
iris['target'] = iris['target'][indices]

iris['target'] = (iris['target'] == 0).astype(int)
iris

split_size = 0.8
split_index = int(np.floor(len(iris['data']) * split_size))
train = {}
train['data'] = iris['data'][:split_index]
train['target'] = iris['target'][:split_index]

test = {}
test['data'] = iris['data'][split_index:]
test['target'] = iris['target'][split_index:]
