from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, GRU, Dropout
import MyCallbacks
from keras import regularizers
import HelpFunctions as hp
import random as rn
import numpy as np
from math import exp
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import matplotlib.pyplot as plt
get_bin = lambda x, n: format(x, 'b').zfill(n)
def shuffle(a,b):
    perm = np.random.permutation(Y.shape[0])
    return a[perm], b[perm]
####load data
dataset = np.loadtxt("num_of_layers_3_rand_units_rand_act_for_all.txt", delimiter=" ")[0:100]
time_distribution_steps = 10
number_of_bits_per_layer = 9
dimensionality = time_distribution_steps*number_of_bits_per_layer
X = np.zeros((dataset.shape[0], time_distribution_steps*number_of_bits_per_layer))
for i in range(0,dataset.shape[0]):
    bit = get_bin(int(dataset[i,0]), time_distribution_steps*number_of_bits_per_layer)
    bit_count = 0
    for steps in range(0, time_distribution_steps*number_of_bits_per_layer):
        X[i, steps] = int(bit[steps])
        bit_count += 1


Y = dataset[:,1:7]
X, Y = shuffle(X, Y)
test_samples = 2
x,x_test, y, y_test = X[0:-test_samples,:],X[-test_samples:-1,:],Y[0:-test_samples,:],Y[-test_samples:-1,:]

def base_model(input):
    layer = Dense(dimensionality,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input)
    # layer = Dropout(0.05)(layer)
    layer = Dense(80,activation='relu',kernel_regularizer=regularizers.l2(0.001) )(layer)
    # layer = Dropout(0.05)(layer)
    model = Model(inputs=input, outputs=layer)
    model.compile(loss='mse', optimizer='nadam', metrics=[])
    return model, layer
def model_for_decoder(input, base):
    x = base(input)
    layer = Dense(80,activation='relu', kernel_regularizer=regularizers.l2(0.001))(x) # Get the last output of the GRU and repeats it
    # layer = Dropout(0.05)(layer)
    # layer = Dense(80, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    # layer = Dense(80, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    output1 = Dense(int(dimensionality),activation='tanh')(layer)
    model = Model(inputs=input,outputs=output1)
    model.compile(loss='mse', optimizer='nadam', metrics=[])
    return model, output1
def model_for_for_values(input, base):
    x = base(input)
    layer = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    layer = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(layer)
    output2 = Dense(6, activation='linear', name='main_output')(layer)
    model = Model(inputs=[input], outputs=[output2])
    model.compile(loss='mse', optimizer='nadam', metrics=[])
    return model, output2
def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model, model2, x, y, dict, batch_size = 10):
    len_of_data = x.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    for i in range(0, int(len_of_data/ batch_size) + 1):
        if (10 + cur_line)<= len_of_data:
            model_loss.append(model.train_on_batch(x[cur_line:(cur_line + 10)], y[cur_line:(cur_line + 10)]))
            model2_loss.append(model2.train_on_batch(x[cur_line:(cur_line + 10)], x[cur_line:(cur_line + 10)], class_weight=dict))
            model2_loss.append(
                model2.train_on_batch(x[cur_line:(cur_line + 10)], x[cur_line:(cur_line + 10)], class_weight=dict))
        else:
            model_loss.append(model.train_on_batch(x[cur_line:len_of_data], y[cur_line:len_of_data]))
            model2_loss.append(model2.train_on_batch(x[cur_line:len_of_data], x[cur_line:len_of_data],class_weight = dict))
            model2_loss.append(
                model2.train_on_batch(x[cur_line:len_of_data], x[cur_line:len_of_data], class_weight=dict))
        print("Epoch #{}: model Loss: {}, model2 Loss: {}".format(epoch + 1, model_loss[-1], model2_loss[-1]))

def pretrain_on_epoch(model2, x, y, dict, batch_size=1):
    len_of_data = x.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    for i in range(0, int(len_of_data / batch_size) + 1):
        if (10 + cur_line) <= len_of_data:
            model2_loss.append(
                model2.train_on_batch(x[cur_line:(cur_line + 10)], x[cur_line:(cur_line + 10)]))
        else:
            model2_loss.append(
                model2.train_on_batch(x[cur_line:len_of_data], x[cur_line:len_of_data]))
        # print("Epoch #{}: , model2 Loss: {}".format(epoch + 1, model2_loss[-1]))

##First partial model
input = Input(shape=(90,))
base_m, base_m_out = base_model(input)

##Two full models
input = Input(shape=(90,))
model_for_decode, model_for_decode_out = model_for_decoder(input, base_m)
input = Input(shape=(90,))
model_to_eval, model_to_eval_out = model_for_for_values(input, base_m)


dict= {}
for i in range(0,10):
    for lay in range(0,9):
        if lay == 7 or lay==8:
            dict[i*9 + lay] = 100
        else:
            dict[i * 9 + lay] = 2**(6 - lay)
# ###First fit model_to_predict
# model_to_eval.fit(x,y, epochs=200, batch_size=10)
#
# #Secondly the encoder
# set_trainable(base_m, False)
# model_for_decode.fit(x,x, epochs=200, batch_size=5)
# for epoch in range(0,100):
#     pretrain_on_epoch(model_for_decode,x,y,dict,1)
#     print('####################################',model_for_decode.test_on_batch(x_test, x_test ))

for epoch in range(0,150):
    train_on_epoch(model_to_eval, model_for_decode, x, y,dict, 1)
    print('####################################', model_for_decode.test_on_batch(x_test, x_test))

print(np.round(model_for_decode.predict(x_test)) - np.round(x_test))
print(x_test)
