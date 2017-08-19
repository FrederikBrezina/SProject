import tensorflow as tf
# Creates a graph.
import numpy as np
f = []
g,h = [],[]
f.append(g)
f.append(h)
f[0].append(3)
f[0].append(4)
f[1].append(1)
f[1].append(2)

print(np.array(f)[0,1])
