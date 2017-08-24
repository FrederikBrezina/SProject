import numpy as np
f = np.loadtxt('../error_list.txt')
for i in range(0, f.shape[0]):
    print(f[i])