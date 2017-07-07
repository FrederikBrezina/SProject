import numpy as np
import matplotlib.pyplot as pll
import threading
g = np.zeros((5,1))
class Mythread(threading.Thread):
    def __init__(self, i, h):
        super(Mythread, self).__init__()
        self.i = i
        self.h = h
    def run(self):
        self.h[self.i][0] = self.i
        return self.i
list1 = []
j = 0
for i in range(0,5):
    list1.append(Mythread(i, g))
    list1[-1].start()

for i in range(0, 5):
    list1[i].join()
print(j)


