from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np
import pickle
from keras.layers.wrappers import TimeDistributed

dimension_of_hidden_layers = 90
def base_model(input):
    layer = TimeDistributed(Dense(dimension_of_hidden_layers))(input)
    layer = LSTM(dimension_of_hidden_layers, kernel_regularizer=regularizers.l2(0.01), dropout=0.2, return_sequences=False)(layer)
    layer = BatchNormalization()(layer)
    model = Model(inputs=input, outputs=layer)
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, layer
def model_for_decoder(input, base):
    x = base(input)
    layer = RepeatVector(10)(x) # Get the last output of the GRU and repeats it
    output1 = LSTM(dimension_of_hidden_layers,  kernel_regularizer=regularizers.l2(0.01), dropout=0.2, return_sequences=True, name='lstm_output')(layer)
    output1 = TimeDistributed(Dense(9))(output1)
    model = Model(inputs=input,outputs=output1)
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, output1
def model_for_for_values(input, base):
    x = base(input)
    layer = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    layer = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(layer)
    output2 = Dense(9, activation='linear', name='main_output')(layer)
    model = Model(inputs=[input], outputs=[output2])
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model, output2
def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model, model2, x, y, epoch, batch_size = 10):
    len_of_data = x.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    for i in range(0, int(len_of_data/ batch_size) + 1):
        futur_line = cur_line + batch_size
        if (futur_line)<= len_of_data:
            model_loss.append(model.train_on_batch(x[cur_line:(futur_line)], y[cur_line:(futur_line)]))
            model2_loss.append(model2.train_on_batch(x[cur_line:(futur_line)], x[cur_line:(futur_line)]))
            cur_line = futur_line
        elif (cur_line<len_of_data):
            print('fuck')
            model_loss.append(model.train_on_batch(x[cur_line:len_of_data], y[cur_line:len_of_data]))
            model2_loss.append(model2.train_on_batch(x[cur_line:len_of_data], x[cur_line:len_of_data]))

        print("Epoch #{}: model Loss: {}, model2 Loss: {}".format(epoch + 1, model_loss[-1], model2_loss[-1]))

def pretrain_on_epoch(model2, x, y, batch_size=1):
    len_of_data = x.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    for i in range(0, int(len_of_data / batch_size) + 1):
        futur_line = cur_line + batch_size
        if futur_line <= len_of_data:
            model2_loss.append(
                model2.train_on_batch(x[cur_line:futur_line], x[cur_line:futur_line]))
            cur_line = futur_line
        else:
            model2_loss.append(
                model2.train_on_batch(x[cur_line:len_of_data], x[cur_line:len_of_data]))
        # print("Epoch #{}: , model2 Loss: {}".format(epoch + 1, model2_loss[-1]))

def trained_models(datax, datay):
    input = Input(shape=(10,9))
    base_m, base_m_out = base_model(input)
    ##Two full models
    input = Input(shape=(10,9))
    model_for_decode, model_for_decode_out = model_for_decoder(input, base_m)
    input = Input(shape=(10,9))
    model_to_eval, model_to_eval_out = model_for_for_values(input, base_m)
    # dict= {}
    # for i in range(0,10):
    #     for lay in range(0,9):
    #         if lay == 7 or lay==8:
    #             dict[i*9 + lay] = 100
    #         else:
    #             dict[i * 9 + lay] = 2**(6 - lay)
    # ###First fit model_to_predict
    # model_to_eval.fit(x,y, epochs=200, batch_size=10)
    #
    # #Secondly the encoder
    # set_trainable(base_m, False)
    # model_for_decode.fit(x,x, epochs=200, batch_size=5)
    # for epoch in range(0,100):
    #     pretrain_on_epoch(model_for_decode,x,y,dict,1)
    #     print('####################################',model_for_decode.test_on_batch(x_test, x_test ))
    for epoch in range(0,10):
        train_on_epoch(model_to_eval, model_for_decode, datax, datay, epoch, 1)
    return base_m, model_for_decode
