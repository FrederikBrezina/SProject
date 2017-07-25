import numpy as np
import pprint, pickle
data = np.ones((3,3))
data[0] = 0
data1 = np.zeros((3,1))
data1[0] = 1
print(np.any(np.abs(data1 - data) <= 0.9))

