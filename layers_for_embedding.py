from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import numpy as np

def main():
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
    number_of_measurements_weight_wise = 8
    number_of_measurements = num_of_layers_tot*number_of_measurements_weight_wise + 20
    measurement_array = np.zeros((3,13,number_of_measurements)) #Where last number is number of measurements, numbers before it are iterations
    measurement_array_editted = np.copy(measurement_array)

    number_of_epochs, number_of_training_examples, batch_size = 100, X.shape[0], 10


    #Observe for number of context units
    for i in range(0,3):
        #Observe layers from 5 to 20 units
        for units in range(3,4):
            #Measurement Values
            #Convregence epoch
            convergence_points = []
            #Weight_matrice measurements
            second_order_before_conv_sqrd = np.zeros((num_of_layers_tot, num_of_tries_tot))
            second_order_before_conv_cubic = np.zeros((num_of_layers_tot, num_of_tries_tot))
            second_order_after_conv_sqrd = np.zeros((num_of_layers_tot, num_of_tries_tot))
            second_order_after_conv_cubic = np.zeros((num_of_layers_tot, num_of_tries_tot))

            sum_of_second_order_before, sum_of_absolute_second_order_before, sum_of_second_order_after, diff_at_conv_loss, max_diff_loss, performance_list = [], [], [], [], [], []


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
                weight_variance_history = MyCallbacks.LayersEmbeddingAllMeasurementsThreaded(number_of_epochs, number_of_training_examples, batch_size)
                val_loss_history = MyCallbacks.ValLossHistory()
                loss_history_epochs = MyCallbacks.LossHistoryEpoch()

                #Fit the model
                model.fit(X, Y, epochs=number_of_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                          callbacks=[weight_variance_history, val_loss_history, loss_history_epochs], shuffle=True)
                #Append measured values
                convergence_points.append(hp.convergence_of_NN_val_loss(val_loss_history.losses, 4))
                second_order_before_conv_sqrd[:,num_of_try] = weight_variance_history.second_derivatives[:,0]
                second_order_before_conv_cubic[:,num_of_try] = weight_variance_history.second_derivatives[:, 1]
                second_order_after_conv_sqrd[:,num_of_try] = weight_variance_history.second_derivatives[:, 2]
                second_order_after_conv_cubic[:,num_of_try] = weight_variance_history.second_derivatives[:, 3]
                
                before, before_abs, after, diff_at_conv, max_diff = hp.overfitting_all_values(loss_history_epochs.losses, val_loss_history.losses, convergence_points[-1])
                sum_of_absolute_second_order_before.append(before_abs), sum_of_second_order_before.append(before), sum_of_second_order_after.append(after), diff_at_conv_loss.append(diff_at_conv), max_diff_loss.append(max_diff)
                performance_list.append(hp.lowest_val_loss(val_loss_history.losses))

                #Calculate the standard deviations and put the measurements in the array
            for num_of_layers in range(0, num_of_layers_tot):
                measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 0], measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 1] = hp.mean_and_std(second_order_before_conv_sqrd[num_of_layers][:])
                measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 2], measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 3] = hp.mean_and_std(second_order_before_conv_cubic[num_of_layers][:])
                measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 4], measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 5] = hp.mean_and_std(second_order_after_conv_sqrd[num_of_layers][:])
                measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 6], measurement_array[i][units - 3][num_of_layers * number_of_measurements_weight_wise + 7] = hp.mean_and_std(second_order_after_conv_cubic[num_of_layers][:])



            measurement_array[i][units - 3][num_of_layers_tot*number_of_measurements_weight_wise + 0], measurement_array[i][units - 3][num_of_layers_tot*number_of_measurements_weight_wise + 1] = hp.mean_and_std(convergence_points)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 2], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 3] = hp.mean_and_std(sum_of_second_order_before)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 4], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 5] = hp.mean_and_std(sum_of_absolute_second_order_before)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 6], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 7] = hp.mean_and_std(sum_of_second_order_after)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 8], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 9] = hp.mean_and_std(diff_at_conv_loss)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 10], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 11] = hp.mean_and_std(max_diff_loss)
            measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 12], measurement_array[i][units - 3][num_of_layers_tot * number_of_measurements_weight_wise + 13] = hp.mean_and_std(performance_list)

    for measurement in range(0, measurement_array.shape[2]):
        measurement_array_editted[:][:][measurement] = hp.return_smoothed_data_with_average_std_given(measurement_array[:][:][measurement], avg=0, std=1)

if __name__ == '__main__':
    main()