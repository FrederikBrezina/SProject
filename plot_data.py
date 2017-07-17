import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
dataset = np.loadtxt("multiple_data_points_multiple_units_in_first_layer.txt", delimiter=" ")
print(dataset[0:100,0])
X,Y = dataset[0:100,0],dataset[0:100,1]
Z = dataset[0:100,2]
E = dataset[0:100,3]
S = Z - E
ax.scatter(X,Y,S)
plt.show()
