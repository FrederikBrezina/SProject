import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn

from math import exp

dataset = np.loadtxt("new.txt", delimiter=" ")
X = dataset[0:-101, 0:2]
Y = dataset[0:-101, 2:5]
x_test = dataset[-101:-1, 0:2]
y_test = dataset[-101:-1, 2:5]

# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 3
input = Input(shape=(2,))
layers = Dense(100, activation=c[0])(input)
layers = Dense(20, activation=c[0])(layers)
layers = Dense(20, activation=c[1])(layers)
layers = Dense(20, activation=c[2])(layers)
output = Dense(num_of_ouputs, activation='sigmoid')(layers)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

weight_variance_history = MyCallbacks.WeightVarianceTest(3)

model.fit(X, Y, epochs=1000, batch_size=10, validation_data=(x_test, y_test),
          callbacks=[weight_variance_history], shuffle=True)