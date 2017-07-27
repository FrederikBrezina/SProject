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
X = np.zeros((dataset.shape[0], time_distribution_steps*number_of_bits_per_layer))
for i in range(0,dataset.shape[0]):
    bit = get_bin(int(dataset[i,0]), time_distribution_steps*number_of_bits_per_layer)
    bit_count = 0
    for steps in range(0, time_distribution_steps*number_of_bits_per_layer):
        X[i, steps] = int(bit[steps])
        bit_count += 1


Y = dataset[:,1:7]
X, Y = shuffle(X, Y)
test_samples = 2
x,x_test, y, y_test = X[0:-test_samples,:],X[-test_samples:-1,:],Y[0:-test_samples,:],Y[-test_samples:-1,:]

input = Input(shape=(90,))
layer = Dense(70)(input)
shared_layer = Dense(40, activation='relu')(layer)
layer = Dense(6, activation='sigmoid')(shared_layer)
model = Model(inputs=input, outputs=layer)
model.compile(loss='mse', optimizer='adam', metrics=[])
model.fit(x,y, epochs=20, batch_size=10, validation_data=[])
shared_layer.trainable = False
layer_out = Dense(90, activation='sigmoid')(shared_layer)
model2 = Model(inputs=input, outputs=layer_out)
model2.compile(loss='mse', optimizer='adam', metrics=[])
model.fit(x,x, epochs=250, batch_size=10, validation_data=[])
print(np.round(model.predict(x_test)) - np.round(x_test))
