import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import sklearn.datasets as ds

N = 400
centers = 4
data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
Y = np.array(y)
DATA = np.array(data)
x_train = np.hstack((DATA, Y))

print(x_train.shape)
