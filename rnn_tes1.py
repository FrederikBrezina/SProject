from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, GRU
import MyCallbacks
from keras import regularizers
import HelpFunctions as hp
import random as rn
import numpy as np
from math import exp
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import matplotlib.pyplot as plt
get_bin = lambda x, n: format(x, 'b').zfill(n)
def shuffle(a,b):
    perm = np.random.permutation(Y.shape[0])
    return a[perm], b[perm]
####load data
dataset = np.loadtxt("num_of_layers_3_rand_units_rand_act_for_all.txt", delimiter=" ")[0:600]
time_distribution_steps = 10
number_of_bits_per_layer = 9
X = np.zeros((dataset.shape[0], time_distribution_steps, number_of_bits_per_layer))
for i in range(0,dataset.shape[0]):
    bit = get_bin(int(dataset[i,0]), time_distribution_steps*number_of_bits_per_layer)
    bit_count = 0
    for steps in range(0, time_distribution_steps):
        for bits_per_layer in range(0, number_of_bits_per_layer):
            X[i, steps, bits_per_layer] = int(bit[bit_count])
            bit_count += 1


Y = dataset[:,1:7]
X, Y = shuffle(X, Y)
test_samples = 2
x,x_test, y, y_test = X[0:-test_samples,:,:],X[-test_samples:-1,:,:],Y[0:-test_samples,:],Y[-test_samples:-1,:]

model = Sequential()
model.add(TimeDistributed(Dense(40), input_shape=(10, 9)))
model.add(GRU(40, input_dim=9,  dropout=0.2))
model.add(RepeatVector(10)) # Get the last output of the GRU and repeats it
model.add(GRU(9, return_sequences=True, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=[])
model.fit(x,x, epochs=250, batch_size=10, validation_data=[])
print(np.round(model.predict(x_test)) - np.round(x_test))
