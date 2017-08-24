import numpy as np
from matplotlib import pyplot as plt
f = np.loadtxt('../error_list.txt')
plt.plot(f[:,1],"r-", f[:,2],'b-')
plt.show()
for i in range(0, f.shape[0]):
    print(f[i])