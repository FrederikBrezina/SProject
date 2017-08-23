import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import matplotlib.pyplot as plt

from math import exp


datax, datay = 'X_basic_task.txt', 'Y_basic_task.txt'
X = np.loadtxt(datax, delimiter=" ")[:500]
Y = np.loadtxt(datay, delimiter=" ")[:500]
test_index = 100
x, x_test, y, y_test = X[:-test_index], X[-test_index:], Y[:-test_index], Y[-test_index:]

# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 1
input = Input(shape=(2,))
output = Dense(20, activation='relu')(input)
output = Dense(20, activation='relu')(output)
output = Dense(3, activation='sigmoid')(output)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, Y, epochs=200, batch_size=10, validation_data=(x_test,y_test),
          callbacks=[], shuffle=True)
