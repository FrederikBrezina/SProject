from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import random as rn
import numpy as np
from math import exp

dataset = np.loadtxt("data_for_real1.txt", delimiter=" ")
X = dataset[0:-101, 0:2]
Y = dataset[0:-101, 2:5]
x_test = dataset[-101:-1, 0:2]
y_test = dataset[-101:-1, 2:5]
#Types of layers used
c = ['relu' , 'tanh' , 'sigmoid' ]
num_of_ouputs = 3
num_of_layers_tot = 5
num_of_tries_tot = 5
measurement_array = np.zeros((3,13,20)) #Where last number is number of measurements, numbers before it are iterations

number_of_epochs, number_of_training_examples, batch_size = 100, X.shape[0], 10


#Observe for number of context units
for i in range(0,3):
    #Observe layers from 5 to 20 units
    for units in range(3,16):
        #Measurement Values
        convergence_points = []
        second_order_before_conv_abs_sum = np.zeros((num_of_layers_tot, num_of_tries_tot))
        second_order_before_conv_sum = np.zeros((num_of_layers_tot, num_of_tries_tot))
        second_order_after_conv_abs_sum = np.zeros((num_of_layers_tot, num_of_tries_tot))
        second_order_after_conv_sum = np.zeros((num_of_layers_tot, num_of_tries_tot))
        second_order_before_conv_sqrd_sum = np.zeros((num_of_layers_tot, num_of_tries_tot))
        act_units = int(1.6**(units))
        for num_of_try in range(0,num_of_tries_tot):
            #choosing activation function
            input = Input(shape=(2,))
            layers = Dense(act_units, activation=c[i])(input)
            layers = Dense(40, activation=c[0])(layers)
            layers = Dense(30, activation=c[1])(layers)
            layers = Dense(20, activation=c[2])(layers)
            output = Dense(num_of_ouputs, activation='sigmoid')(layers)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            #Create callback
            weight_variance_history = MyCallbacks.LayersEmbeddingAllMeasurements(number_of_epochs, number_of_training_examples, batch_size)
            val_loss_history = MyCallbacks.ValLossHistory()

            #Fit the model
            model.fit(X, Y, epochs=number_of_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                      callbacks=[weight_variance_history, val_loss_history], shuffle=True)
            #Append measured values
            convergence_points.append(hp.convergence_of_NN_val_loss(val_loss_history.losses, 4))
            second_order_before_conv_abs_sum[:][num_of_try] = weight_variance_history.second_derivatives[:,0]
            second_order_before_conv_sum[:][num_of_try] = weight_variance_history.second_derivatives[:, 1]
            second_order_after_conv_abs_sum[:][num_of_try] = weight_variance_history.second_derivatives[:, 2]
            second_order_after_conv_sum[:][num_of_try] = weight_variance_history.second_derivatives[:, 3]
            second_order_before_conv_sqrd_sum[:][num_of_try] = weight_variance_history.second_derivatives[:, 4]

            #Calculate the standard deviations and put the measurements in the array
        for num_of_layers in range(0, num_of_layers_tot):
            measurement_array[i][units - 3][num_of_layers * 10 + 0], measurement_array[i][units - 3][num_of_layers * 10 + 1] = hp.mean_and_std(second_order_before_conv_abs_sum[num_of_layers][:])
            measurement_array[i][units - 3][num_of_layers * 10 + 2], measurement_array[i][units - 3][num_of_layers * 10 + 3] = hp.mean_and_std(second_order_before_conv_sum[num_of_layers][:])
            measurement_array[i][units - 3][num_of_layers * 10 + 4], measurement_array[i][units - 3][num_of_layers * 10 + 5] = hp.mean_and_std(second_order_after_conv_abs_sum[num_of_layers][:])
            measurement_array[i][units - 3][num_of_layers * 10 + 6], measurement_array[i][units - 3][num_of_layers * 10 + 7] = hp.mean_and_std(second_order_after_conv_sum[num_of_layers][:])
            measurement_array[i][units - 3][num_of_layers * 10 + 8], measurement_array[i][units - 3][num_of_layers * 10 + 9] = hp.mean_and_std(second_order_before_conv_sqrd_sum[num_of_layers][:])


        measurement_array[i][units - 3][num_of_layers_tot*10 + 0], measurement_array[i][units - 3][num_of_layers_tot*10 + 1] = hp.mean_and_std(convergence_points)
