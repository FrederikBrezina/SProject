from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import HelpFunctions as hp


class WeightVarianceHistory(Callback):
    def __init__(self):
        super(WeightVarianceHistory, self).__init__()
        #Temporary parameters
        self.finish_model = []
        self.start_model = []
        self.model_change = []
        self.last_model = []
        self.all_models = []
        self.all_models_average = []
        self.epoch_num = 1
        self.top10=np.zeros((10,5)) #Where first 3 are indeces where to find the average, 4th
                                    #is the abs(average) and the 5th is the variance of that weight
        #Callback parameters
        self.model_change_average = 0
        self.weight_var = [0 , 0] #[0] -> sum of averages of top10 largest weights, [1] -> sum of standard deviations of top10 weights

    def on_train_begin(self, logs={}):
        self.model_change = self.model.get_weights()
        self.start_model = self.model.get_weights()
        self.last_model = self.model.get_weights()
        self.all_models = self.model.get_weights()
        self.all_models_average = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_num += 1
        # Appending all history of weights nad calculating its variance
        self.all_models.append(self.model.get_weights)
        self.finish_model = self.model.get_weights()
        for i1 in range(0, len(self.finish_model)):
            for i2 in range(0, self.finish_model[i1].shape[0]):
                for i3 in range(0, self.finish_model[i1].shape[1]):
                    self.all_models_average[i1][i2][i3] += self.finish_model[i1][i2][i3]

    def on_batch_end(self, batch, logs={}):
        #Calculating the total distance traveled by the weight
        self.finish_model = self.model.get_weights()
        for i1 in range(0,len(self.finish_model)):
            for i2 in range(0, self.finish_model[i1].shape[0]):
                for i3 in range(0, self.finish_model[i1].shape[1]):
                    self.model_change[i1][i2][i3] += abs(self.finish_model[i1][i2][i3] - self.last_model[i1][i2][i3])

        self.last_model=self.last_model

    def on_train_end(self, logs=None):
        self.finish_model=self.model.get_weights()
        unit_count=0
        for i1 in range(0,len(self.finish_model)):
            for i2 in range(0, self.finish_model[i1].shape[0]):
                for i3 in range(0, self.finish_model[i1].shape[1]):
                    unit_count += 1
                    self.model_change[i1][i2][i3] -= self.start_model[i1][i2][i3]
                    self.model_change[i1][i2][i3] /= abs(self.finish_model[i1][i2][i3] - self.start_model[i1][i2][i3])
                    self.model_change_average += self.model_change[i1][i2][i3]
                    self.all_models_average[i1][i2][i3] /= self.epoch_num

        #Average distance traveled for one unit
        self.model_change_average /= unit_count

        #CHoose ten biggest absolute values from matrix and calculate their variance
        for i1 in range(0,len(self.finish_model)):
            for i2 in range(0, self.finish_model[i1].shape[0]):
                for i3 in range(0, self.finish_model[i1].shape[1]):
                    x = 0  #If x stayes 0 than the number is no bigger than any of the numbers
                    for i4 in range(0,self.top10.shape[0]):
                        #starting from the bottom of the array where elements are the lowest
                        if abs(self.all_models_average)>abs(self.top10[-i4-1,3]):
                            x = -i4-1
                    #Move all the subsequent positions by one down
                    for i4 in range(0,abs(x)):
                        # Special case if it is last element
                        if abs(x) == 1:
                            self.top10[-1][3] = self.all_models_average[i1][i2][i3]
                            self.top10[-1][0] = i1
                            self.top10[-1][1] = i2
                            self.top10[-1][2] = i3
                        #Standard other cases
                        if i4 != abs(x)-1:
                            self.top10[-1-i4][3] = self.top10[-2-i4][3]
                            self.top10[-1-i4][0] = self.top10[-2-i4][0]
                            self.top10[-1-i4][1] = self.top10[-2-i4][1]
                            self.top10[-1-i4][2] = self.top10[-2-i4][2]
                        #Putting the new value at the right place
                        if i4 ==abs(x) -1:
                            self.top10[x][3] = self.all_models_average[i1][i2][i3]
                            self.top10[x][0] = i1
                            self.top10[x][1] = i2
                            self.top10[x][2] = i3

        #Calculate the standard deviation of the largest abs means
        for x1 in range(0,self.top10.shape[0]):
            for x2 in range(0, len(self.all_models)):
                self.top10[x1][4] += (self.top10[x1][3] - self.all_models[x2][self.top10[x1][0]][self.top10[x1][1]][self.top10[x1][2]])**2

            self.top10[x1][4] = self.top10[x1][4]**0.5
            self.weight_var[0] += abs(self.top10[x1][3])
            self.weight_var[1] += self.top10[x1][4]



class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

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

class WeightVarianceTest(Callback):
    def __init__(self, number_of_context_layers=3):
        super(WeightVarianceTest, self).__init__()
        self.list = []

        self.batch_list = []
        #measure 3 units per layer
        for i in range(0,15):
            g = []
            self.list.append(g)


    def on_batch_end(self, batch, logs=None):


        for i in range(0,5):
            self.list[(i * 3 + 0)].append(self.model.layers[i+1].get_weights()[0][0][0])
            self.list[i * 3 + 1].append(self.model.layers[i+1].get_weights()[0][0][1])
            self.list[i * 3 + 2].append(self.model.layers[i+1].get_weights()[0][1][0])

    def on_train_end(self, logs=None):
        #Smoothing out data
        for element in range(0,len(self.list)):
            self.list[element] = hp.smooth_the_data_moving_average(self.list[element], 70)
        for batch_num in range(0, len(self.list[0])):
            self.batch_list.append(batch_num)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.batch_list[0:-2], hp.second_order_derivate(self.list[1]), 'r-', self.batch_list[0:-2], hp.second_order_derivate(self.list[4]), 'b-', self.batch_list[0:-2], hp.second_order_derivate(self.list[7]), 'y-', self.batch_list[0:-2], hp.second_order_derivate(self.list[10]), 'g-', self.batch_list[0:-2], hp.second_order_derivate(self.list[13]), 'k-')
        plt.subplot(212)
        plt.plot(self.batch_list, self.list[1], 'r-', self.batch_list,
                 self.list[4], 'b-', self.batch_list, self.list[7],
                 'y-', self.batch_list, self.list[10], 'g-', self.batch_list,
                 self.list[13], 'k-')

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


