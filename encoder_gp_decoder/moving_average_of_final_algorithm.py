import numpy as np
import pickle
import matplotlib.pyplot as plt
pkl_file = open('arch_list.pkl', 'rb')
data1 = pickle.load(pkl_file)
print(data1)
f = np.array(np.loadtxt("performance1.txt"))
running_sum_list = []
ran = 100
for i in range(0, data1.shape[0] - ran + 1):
    run_sum = 0
    for i2 in range(ran):
        run_sum+=data1[i+i2]
    run_sum/=ran
    running_sum_list.append(run_sum)
plt.plot(f)
plt.show()