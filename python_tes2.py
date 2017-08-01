import numpy as np
import pprint, pickle
dict= {}
for i in range(0,10):
    for lay in range(0,9):
        if lay == 7 or lay==8:
            dict[i*9 + lay] = 3
        else:
            dict[i * 9 + lay] = 1

print(dict[7])
