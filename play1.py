import numpy as np
dataset = np.loadtxt('new.txt', delimiter=" ")
x = dataset[:, 0:2]
y = dataset[:, 2:5]
np.savetxt('X_basic_task.txt', x, delimiter=" ")
np.savetxt('Y_basic_task.txt', y, delimiter=" ")