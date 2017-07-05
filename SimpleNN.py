import numpy as np
import matplotlib.pyplot as plt
dataset = np.loadtxt("units_of_first_layer.txt", delimiter=" ")

X = dataset[:, 0:1].tolist()
del X[0]
print(X)
