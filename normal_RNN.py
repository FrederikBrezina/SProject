from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector
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
dataset = np.loadtxt("num_of_layers_3_sequential.txt", delimiter=" ")
time_distribution_steps = 3
number_of_bits_per_layer = 9
X = np.zeros((dataset.shape[0], time_distribution_steps, number_of_bits_per_layer))
for i in range(0,dataset.shape[0]):
    bit = get_bin(int(dataset[i,0]), time_distribution_steps*number_of_bits_per_layer)
    bit_count = 0
    for steps in range(0, time_distribution_steps):
        for bits_per_layer in range(0, number_of_bits_per_layer):
            X[i, steps, bits_per_layer] = bit[bit_count]
            bit_count += 1


Y = dataset[:,1:7]
X, Y = shuffle(X, Y)

x,x_test, y, y_test = X[0:-20,:,:],X[-20:-1,:,:],Y[0:-20,:],Y[-20:-1,:]

model = Sequential()
model.add(LSTM(64, input_dim=9, kernel_regularizer=regularizers.l2(0.01), dropout=0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(6, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=[])
model.fit(x,y, epochs=500, batch_size=10, validation_data=[])
plt.plot(model.predict(x_test)[:,0], 'b-', y_test[:,0], 'r-')
plt.show()