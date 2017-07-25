import numpy as np
np.random.seed(7)
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import MyCallbacks
import random
import HelpFunctions as hp
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import gp_bayes as bayes
dataset = np.loadtxt('num_of_layers_3_rand_units_rand_act_for_all.txt', delimiter=" ")
get_bin = lambda x, n: format(x, 'b').zfill(n)
number_of_bits_per_layer = 9
number_of_bits = int((dataset[0,-1] - 1)*number_of_bits_per_layer)
Y = dataset[:,1:-2]
X = np.zeros((dataset.shape[0],number_of_bits))
c = ['relu', 'tanh', 'sigmoid']
def shuffle(a,b):
    perm = np.random.permutation(Y.shape[0])
    return a[perm], b[perm]
for i in range(0,dataset.shape[0]):
    X[i,:] = get_bin(int(dataset[i,0]), number_of_bits)

X, Y  = shuffle(X,Y)

x,x_test, y, y_test = X[0:-101],X[-101:-1],Y[0:-101],Y[-101,-1]

##The approximatelly same for all regressions
#############################################
def loss_nn_dense(args):
    depth = int(len(args)/2)
    model = Sequential()
    for layer in range(0, depth):
        if layer ==0:
            model.add(Dense(int(args[layer*2]), activation=c[int(args[layer*2 + 1])], input_shape = (2,)))
        else:
            model.add(Dense(int(args[layer*2]), activation=c[int(args[layer*2 + 1])]))
    model.add(Dense(3, activation='tanh'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    acc_his = MyCallbacks.AccHistoryEpochTest()

    model.fit(x, y, epochs=200, batch_size=16, validation_data=(x_test, y_test),
              callbacks=[acc_his], shuffle=True)

    return min(acc_his.losses_val_losses)
