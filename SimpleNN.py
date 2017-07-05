import numpy as np
z = []
f = [1 ,2]
z.append(f)
z.append(f)
print(z[1][1])
np.savetxt('test.txt', z, delimiter=' ')
print(np.loadtxt('test.txt', delimiter=' '))
