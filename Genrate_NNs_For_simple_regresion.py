import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import matplotlib.pyplot as plt

from math import exp

dataset = np.loadtxt("new_training_for_simple_reg.txt", delimiter=" ")
X = dataset[0:-101, 0:2]
Y = dataset[0:-101, 2:5]
x_test = dataset[-101:-1, 0:2]
y_test = dataset[-101:-1, 2:5]

# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 3
layer_history = []
for units in range(14,100):
    act_units = 30*units
    input = Input(shape=(2,))
    layers = Dense(act_units, activation=c[0])(input)
    layers = Dense(30, activation=c[1])(layers)
    layers = Dense(20, activation=c[1])(layers)
    output = Dense(num_of_ouputs, activation='sigmoid')(layers)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_his = MyCallbacks.AccHistoryEpochTest()

    model.fit(X, Y, epochs=300, batch_size=15, validation_data=(x_test, y_test),
              callbacks=[acc_his], shuffle=True)
    layer_history.append([act_units, max(acc_his.losses_val)])
    print(act_units)
np.savetxt('units_of_first_layer_from_420_to_3000.txt', layer_history, delimiter=" ")
