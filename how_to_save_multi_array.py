import numpy as np

data = np.ones((3,3,3))
data[0,0,0] = 0.1283817283798123901
# Write the array to disk
import pickle


output = open('data.pkl', 'wb')
pickle.dump(data, output)
output.close()


pkl_file = open('data.pkl', 'rb')

data1 = pickle.load(pkl_file)
print(data1)

pkl_file.close()
