from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import numpy as np
from math import exp

dataset = np.loadtxt("new.txt", delimiter=" ")
X = dataset[0:-101,0:2]
Y = dataset[0:-101,2:5]
x_test = dataset[-101:-1,0:2]
y_test = dataset[-101:-1,2:5]

#Types of layers used
c = ['relu' , 'tanh' , 'sigmoid' ]
num_of_ouputs = 3

#Observe for number of context units
for context in range (1,8):
    context_units = 2^(context)
    for i in range(0,3):
        layers,input = Input(shape=(2,))

        #Observe layers from 5 to 20 units
        for units in range(5,20):
            #choosing number of units in layers
            rand = rn.random()*4 + 1
            units = round(exp(rand))
            #choosing activation function

            layers = Dense(units, activation=c[i])(layers)
            layers = Dense(context_units, activation=c[0])(layers)
            layers = Dense(context_units, activation=c[1])(layers)
            layers = Dense(context_units, activation=c[2])(layers)
            output = Dense(num_of_ouputs, activation='sigmoid')(layers)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


            weight_variance_history = MyCallbacks.WeightVarianceTest()

            model.fit(X, Y, epochs=150, batch_size=10, validation_data=(x_test, y_test),
                      callbacks=[weight_variance_history], shuffle=True)