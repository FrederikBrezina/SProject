from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import HelpFunctions as hp
import HelpFunctionsThreading as hpt


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class LossHistoryEpoch(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

class AccHistoryEpochTest(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.losses_val = []
        self.losses_val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('acc'))
        self.losses_val.append(logs.get('val_acc'))
        self.losses_val_losses.append(logs.get('val_loss'))

class ValLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))

class ValAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_acc'))

class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lossesEpoch = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs=None):
        self.lossesEpoch.append(logs.get('acc'))

class LayersEmbeddingAllMeasurements(Callback):
    """
    returns:
             sum of second derivatives of weights before convergence
             sum of second derivative of weights after convergence
    """
    def __init__(self, number_of_epochs = 300, number_of_train_points = 299, batch_size = 20):
        super(LayersEmbeddingAllMeasurements, self).__init__()
        self.batch_num = 0
        self.list = []
        self.list_second_derivative = []
        self.losses_val =[]
        self.number_of_batches_per_epoch = int(number_of_train_points/batch_size)
        self.num_of_time_steps = int(number_of_epochs * self.number_of_batches_per_epoch)




    def on_train_begin(self, logs=None):
        #Create the list of 3d array
        for i in range(1,len(self.model.layers)):
            arr = np.zeros((self.num_of_time_steps, self.model.layers[i].get_weights()[0].shape[0], self.model.layers[i].get_weights()[0].shape[1]))
            self.list.append(arr)
        self.second_derivatives = np.zeros((len(self.model.layers) - 1,2))

    def on_epoch_end(self, epoch, logs=None):
        self.losses_val.append(logs.get('val_loss'))

    def on_batch_end(self, batch, logs=None):
        if self.list[0].shape[0] > self.batch_num: #Just to make sure we calculated the banch_number correctly
            for layers in range(0,len(self.list)):
                self.list[layers][self.batch_num] = self.model.layers[layers+1].get_weights()[0]
        self.batch_num += 1


    def on_train_end(self, logs=None):
        #Calculate convergence point
        convergence_time_step = (hp.convergence_of_NN_val_loss(self.losses_val,4) * self.number_of_batches_per_epoch) - 1
        #Smoothing out data
        for layers in range(0, len(self.list)):
            second_derivative_sum_before_conv = 0
            second_derivative_sum_after_conv = 0
            for axis_0 in range(0, self.list[layers].shape[1]):
                for axis_1 in range(0, self.list[layers].shape[2]):
                    temp_before_conv = hp.smooth_the_data_moving_average(self.list[layers][0:convergence_time_step,axis_0,axis_1], 240)
                    temp2_before_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_before_conv), 50)
                    second_derivative_sum_before_conv += np.sum(np.absolute(temp2_before_conv))
                    temp_after_conv = hp.smooth_the_data_moving_average(self.list[layers][convergence_time_step:-1,axis_0,axis_1], 240)
                    temp2_after_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_after_conv), 50)
                    second_derivative_sum_after_conv += np.sum(np.absolute(temp2_after_conv))
            self.second_derivatives[layers][0], self.second_derivatives[layers][1] = second_derivative_sum_before_conv, second_derivative_sum_after_conv

class LayersEmbeddingAllMeasurementsThreaded(Callback):
    """
    returns:
             sum of second derivatives of weights before convergence
             sum of second derivative of weights after convergence
    """
    def __init__(self, number_of_epochs = 300, number_of_train_points = 299, batch_size = 20):
        super(LayersEmbeddingAllMeasurementsThreaded, self).__init__()
        self.batch_num = 0
        self.list = []
        self.losses_val =[]
        self.number_of_batches_per_epoch = int(number_of_train_points/batch_size)
        self.num_of_time_steps = int(number_of_epochs * self.number_of_batches_per_epoch)

    def on_train_begin(self, logs=None):
        #Create the list of 3d array
        for i in range(1,len(self.model.layers)):
            arr = np.zeros((self.num_of_time_steps, self.model.layers[i].get_weights()[0].shape[0], self.model.layers[i].get_weights()[0].shape[1]))
            self.list.append(arr)

        self.second_derivatives = np.zeros((len(self.model.layers) - 1,5))

    def on_epoch_end(self, epoch, logs=None):
        self.losses_val.append(logs.get('val_loss'))

    def on_batch_end(self, batch, logs=None):
        if self.list[0].shape[0] > self.batch_num: #Just to make sure we calculated the banch_number correctly
            for layers in range(0,len(self.list)):
                self.list[layers][self.batch_num] = self.model.layers[layers+1].get_weights()[0]
        self.batch_num += 1

    def on_train_end(self, logs=None):
        #Calculate convergence point
        convergence_time_step = (hp.convergence_of_NN_val_loss(self.losses_val,4) * self.number_of_batches_per_epoch) - 1
        #Calculating the data
        hpt.calculate(self.second_derivatives, self.list , convergence_time_step)




class WeightVarianceTest(Callback):
    def __init__(self, number_of_context_layers=3):
        super(WeightVarianceTest, self).__init__()
        self.list = []
        self.list_second_derivative = []

        self.batch_list = []
        self.batch_list2 = []
        self.batch_num = 0
        #measure 3 units per layer
        for i in range(0,15):
            g = []
            self.list.append(g)

    def on_batch_end(self, batch, logs=None):
        self.batch_num += 1

        for i in range(0,5):
            self.list[(i * 3 + 0)].append(self.model.layers[i+1].get_weights()[0][0][0])
            self.list[i * 3 + 1].append(self.model.layers[i+1].get_weights()[0][0][1])
            self.list[i * 3 + 2].append(self.model.layers[i+1].get_weights()[0][1][0])

    def on_train_end(self, logs=None):
        print(self.batch_num)
        #Smoothing out data
        for element in range(0, len(self.list)):
            self.list[element] = hp.smooth_the_data_moving_average(self.list[element], 240)
        for element in range(0,len(self.list)):
            self.list_second_derivative.append(hp.smooth_the_data_moving_average(hp.second_order_derivate(self.list[element]), 240))
        for batch_num in range(0, len(self.list[0])):
            self.batch_list.append(batch_num)
        for batch_num in range(0, len(self.list_second_derivative[0])):
            self.batch_list2.append(batch_num)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.batch_list2, self.list_second_derivative[1], 'r-',
                 self.batch_list2, self.list_second_derivative[4], 'b-',
                 self.batch_list2, self.list_second_derivative[7], 'y-',
                 self.batch_list2, self.list_second_derivative[10], 'g-',
                 self.batch_list2, self.list_second_derivative[13], 'k-')
        plt.subplot(212)
        plt.plot(self.batch_list, self.list[1], 'r-',
                 self.batch_list, self.list[4], 'b-',
                 self.batch_list, self.list[7], 'y-',
                 self.batch_list, self.list[10], 'g-',
                 self.batch_list, self.list[13], 'k-')
        plt.show()



class WeightVarianceHistoryLayers(Callback):
    def __init__(self):
        super(WeightVarianceHistoryLayers, self).__init__()
        self.weight_history = []


    def on_train_begin(self, logs={}):
        for layer in range(0,len(self.model.layers)-1):
            set_list = []
            self.weight_history.append(set_list)

    def on_batch_end(self, batch, logs={}):
        for layer in range(1, len(self.model.layers)):
            self.weight_history[layer-1].append(self.model.layers[layer].get_weights()[0])


