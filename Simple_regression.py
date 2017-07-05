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
X = dataset[0:-11, 0:1]
Y = dataset[0:-11, 1:2]
x_test = dataset[-11:-1, 0:1]
y_test = dataset[-11:-1, 1:2]

# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 1
input = Input(shape=(1,))
output = Dense(num_of_ouputs, activation='linear')(input)
model = Model(inputs=input, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

acc_his = MyCallbacks.AccHistoryEpochTest()

model.fit(X, Y, epochs=500, batch_size=10, validation_data=(x_test, y_test),
          callbacks=[acc_his], shuffle=True)
print(hp.convergence_of_NN_val_loss(acc_his.losses_val_losses,10))
plt.plot(acc_his.losses, 'b-', acc_his.losses_val, 'r-')
plt.show()