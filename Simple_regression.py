import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import matplotlib.pyplot as plt

from math import exp

dataset = np.loadtxt("units_of_first_layer.txt", delimiter=" ")


X = dataset[:, 0:1].tolist()
Y = dataset[:, 1:2].tolist()
x_test = []
y_test = []
for val in range(0,7):
    ra = np.random.randint((len(X) - 1))
    x_test.append(X[ra]), y_test.append(Y[ra])
    del X[ra]
    del Y[ra]

# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 1
input = Input(shape=(1,))
output = Dense(num_of_ouputs, activation='linear')(input)
model = Model(inputs=input, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


model.fit(X, Y, epochs=500, batch_size=10, validation_data=(x_test,y_test),
          callbacks=[], shuffle=True)