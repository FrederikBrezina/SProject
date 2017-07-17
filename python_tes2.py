import numpy as np
f = '0010101'
d = '010101'
g = [[int(f,base=2), 3], [int(d,base=2), 4]]
np.savetxt('testy1.txt', g, delimiter=" ")
print(np.loadtxt('testy1.txt'))