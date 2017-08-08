import numpy as np
import matplotlib.pyplot as plt
X = np.loadtxt("loss.txt", )
f = []
for i in X:
    if i < 2:
        f.append(i)
plt.plot(f)

plt.show()
print(f)