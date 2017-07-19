import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import matplotlib.pyplot as plt
import tensorflow as tf
from math import exp
from keras.layers import LSTM
from keras import backend as K
c = ['relu', 'tanh', 'sigmoid']
dataset = np.loadtxt("new_training_for_simple_reg.txt", delimiter=" ")
X = dataset[0:-101, 0:2]
Y = dataset[0:-101, 2:5]
x_test = dataset[-101:-1, 0:2]
y_test = dataset[-101:-1, 2:5]

input1 = Input(shape=(2,), name='input1')
input2 = Input(shape=(2,), name='input2')
layers2 = Dense(20, activation=c[1])(input2)
output2 = Dense(3, activation='sigmoid')(layers2)
layers = Dense(30, activation=c[0])(input1)
layers = Dense(30, activation=c[1])(layers)
layers = Dense(20, activation=c[1])(layers)
output = Dense(3, activation='sigmoid')(layers)
model = Model(inputs=[input1, input2], outputs=[output, output2])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
acc_his = MyCallbacks.AccHistoryEpochTest()

model.fit([X,X],[Y, Y], epochs=300, batch_size=15, validation_data=([x_test, x_test], [y_test, y_test]),
          callbacks=[acc_his], shuffle=True)
