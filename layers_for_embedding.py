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

measurement_array = np.zeros((3,13,10)) #Where last number is number of measurements

#Observe for number of context units
for i in range(0,3):
    #Observe layers from 5 to 20 units
    for units in range(3,16):
        #Measurement Values
        convergence_points = []
        second_order_before_conv = []
        second_order_after_conv = []
        act_units = int(1.6**(units))
        for num_of_try in range(0,5):
            #choosing activation function
            layers, input = Input(shape=(2,))
            layers = Dense(act_units, activation=c[i])(layers)
            layers = Dense(40, activation=c[0])(layers)
            layers = Dense(30, activation=c[1])(layers)
            layers = Dense(20, activation=c[2])(layers)
            output = Dense(num_of_ouputs, activation='sigmoid')(layers)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            #Create callback
            weight_variance_history = MyCallbacks.LayersEmbeddingAllMeasurements()

            #Fit the model
            model.fit(X, Y, epochs=150, batch_size=10, validation_data=(x_test, y_test),
                      callbacks=[weight_variance_history], shuffle=True)
            #Append measured values
            convergence_points.append(weight_variance_history.convergence_time_step)
            second_order_before_conv.append(weight_variance_history.second_derivative_sum_before_conv)
            second_order_after_conv.append(weight_variance_history.second_derivative_sum_after_conv)

        #Calculate the standard deviations and put the measurements in the array
        measurement_array[i][units - 3][0] , measurement_array[i][units - 3][1] = hp.mean_and_std(second_order_before_conv)
        measurement_array[i][units - 3][2], measurement_array[i][units - 3][3] = hp.mean_and_std(second_order_after_conv)
        measurement_array[i][units - 3][4], measurement_array[i][units - 3][5] = hp.mean_and_std(convergence_points)
