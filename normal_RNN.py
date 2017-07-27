from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, Flatten
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

def base_model(input):
    layer = LSTM(9, kernel_regularizer=regularizers.l2(0.01), dropout=0.2, return_sequences=False)(input)
    model = Model(inputs=input, outputs=layer)
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, layer
def model_for_decoder(input, base):
    x = base(input)
    layer = RepeatVector(3)(x) # Get the last output of the GRU and repeats it
    output1 = LSTM(9,  kernel_regularizer=regularizers.l2(0.01), dropout=0.2, return_sequences=True, name='lstm_output')(layer)
    model = Model(inputs=input,outputs=output1)
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, output1
def model_for_for_values(input, base):
    x = base(input)
    layer = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    layer = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(layer)
    output2 = Dense(6, activation='linear', name='main_output')(layer)
    model = Model(inputs=[input], outputs=[output2])
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, output2
def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

##First partial model
input = Input(shape=(3,9))
base_m, base_m_out = base_model(input)

##Two full models
input = Input(shape=(3,9))
model_for_decode, model_for_decode_out = model_for_decoder(input, base_m)
input = Input(shape=(3,9))
model_to_eval, model_to_eval_out = model_for_for_values(input, base_m)

###First fit model_to_predict
model_to_eval.fit(x,y, epochs=20, batch_size=10)

#Secondly the encoder
set_trainable(base_m, False)
model_for_decode.fit(x,x, epochs=20, batch_size=10)

print(np.round(model_for_decode.predict(x_test)) - np.round(x_test))

