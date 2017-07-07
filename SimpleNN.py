import numpy as np
import matplotlib.pyplot as pll
l = []

g = np.zeros((2,1))
l.append(g)
g = np.ones((2,1))
l.append(g)
print(l[:,0])
s = []
s.append(l[:,0])
s.append(l[:,1])
print(s[:,0])