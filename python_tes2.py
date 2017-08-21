import numpy as np
import pprint, pickle
import pickle
from matplotlib import pyplot as plt
pkl_file = open('loss.txt', 'rb')

data1 = pickle.load(pkl_file)
data1 = np.array(data1)

for i in range(0, data1.shape[0]):
    print(data1[i])

plt.plot(data1)
plt.show()

