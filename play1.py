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

from keras.models import Sequential
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate

def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    print(sh[0])
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    """Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name='test')


    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])

dataset = np.loadtxt("new_training_for_simple_reg.txt", delimiter=" ")
X = dataset[0:100, 0:2]
Y = dataset[0:100, 2:5]
x_test = dataset[-10000:-1, 0:2]
y_test = dataset[-10000:-1, 2:5]
model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(2)))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='sigmoid'))

model = to_multi_gpu(model, n_gpus=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=15, batch_size=16, validation_data=(x_test, y_test),
              callbacks=[], shuffle=True)