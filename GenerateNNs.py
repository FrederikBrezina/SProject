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
c = ['relu' 'tanh']
num_of_ouputs = 3

#min and maximum number of layers
max_layer = 8
min_layer = 1 #Not including the sigmoid layer

#Number of trained examples of each NN
number_of_each_NN = 10

#Convergence of NNs based on val_loss, number of epochs before the best model to look for min
convergence_rate = 4

#Structure of models
NNs = np.zeros((1000,max_layer,2))

#Feedable outputs from the trained NNs
#[0] -> number of epochs where the model converged (mean)
#[1] -> standard deviation of the epochs convergence
#[2] -> robustness to advesarial examples
#[3] -> standard deviation of the one above
#[4] ->
#[5] ->
#[6] ->
#[7] ->
#[8] ->
#[9] ->
#[10] -> mean rate of convergence (as a epoch series) where rate is probability of that epoch
#[11] -> standard deviation of above
#[12] -> mean standard deviation of each of convergance as [10]
#[13] -> standard deviation of above
#[14] -> mean rate of convergence but emphasis taken on peaks (epochs as time series and rate is expectation)
#[15] -> standard deviation of above
#[16] -> mean standard deviation of rate of convergence but..[14]
#[17] -> standard deviation of above
#[18] -> epoch at which overfitting started to happen (mean)
#[19] -> standard dev of above
#[20] -> rate of overfitting (measured from where overfitting started to end of training
#[21] -> standard deviation of above
#[22] -> mean performance
#[23] -> standard deviation of performance
Results = np.zeros((1000,24))

#Choose the neural network, make sure that no two are same
for i in range(0,1000):
    layers,input = Input(shape=(2,))
    #At least 1 layers +1 sigmoid
    length = round(rn.random()*(max_layer-min_layer)+min_layer)

    #Stack layers on top of each other
    for depth in range(0,length):
        #choosing number of units in layers
        rand = rn.random()*4 + 1
        units = round(exp(rand))
        #choosing activation function
        c1 = rn.choice(c)
        layers = Dense(units, activation=c1)(layers)
    output = Dense(num_of_ouputs, activation='sigmoid')(layers)

    #Watch variables
    loss_hist_list = []
    val_loss_hist_list = []
    acc_hist_list = []
    val_acc_hist_list = []
    convergence_list = []
    rate_of_val_acc = []
    epoch_acc_hist_list = []


    #Train 10 times each of the network
    for attempt in range(0,number_of_each_NN):

        model = Model(inputs=input,outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Callbacks
        loss_hist = MyCallbacks.LossHistory()
        val_loss_hist = MyCallbacks.ValLossHistory()
        acc_hist = MyCallbacks.AccHistory()
        val_acc_hist = MyCallbacks.ValAccHistory()


        model.fit(X, Y, epochs=150, batch_size=10, validation_data=(x_test, y_test), callbacks=[loss_hist, val_loss_hist, acc_hist, val_acc_hist, weight_variance_history])

        #Getting the statistics out
        convergence_list.append(hp.convergence_of_NN_val_loss(val_loss_hist.losses,convergence_rate))
        loss_hist_list.append(loss_hist.losses)
        val_loss_hist_list.append(val_loss_hist.losses)
        acc_hist_list.append(acc_hist.losses)
        val_acc_hist_list.append(val_acc_hist.losses)
        rate_of_val_acc.append(hp.rate_of_list(val_acc_hist.losses[0:convergence_list[-1]])) #Take into account only up to convergence, some models do not have overfitting
        epoch_acc_hist_list.append(acc_hist.lossesEpoch)

    #Putting to feedable format to RNN
    Results[i][0],Results[i][1] = hp.mean_and_std(convergence_list)
    Results[i][2], Results[i][3] =
    Results[i][4], Results[i][5] =
    Results[i][6], Results[i][7] =
    Results[i][8], Results[i][9] =
    Results[i][10], Results[i][11], Results[i][12], Results[i][13], Results[i][14], Results[i][15], Results[i][16], Results[i][17] = hp.mean_and_std_of_list_of_lists(rate_of_val_acc)
    Results[i][18], Results[i][19], Results[i][20], Results[i][21] = hp.avg_and_std_of_overfitting(acc_hist_list, val_acc_hist_list, convergence_list)
    Results[i][22], Results[i][23] = hp.mean_and_std_of_performance_list_of_lists(val_acc_hist_list)




