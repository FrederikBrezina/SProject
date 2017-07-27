import dill
import os
import random
import numpy as np
import pickle



get_bin = lambda x, n: format(x, 'b').zfill(n)

s = '0101'
s_list = []
s2 = s
s = s+s
s += '01'
s_list.append(s)
s_list.append(s2)
output = open('data3.pkl', 'wb')
pickle.dump(s_list, output)
output.close()

pkl_file = open('data3.pkl', 'rb')

data1 = pickle.load(pkl_file)
print(data1)
f = 2
f+= int(data1[0][0])
print(data1)
print(f)

pkl_file.close()

