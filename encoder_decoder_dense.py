import pickle
from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, concatenate, Flatten

from keras import regularizers
from keras.layers.local import LocallyConnected1D
import numpy as np

import sklearn.gaussian_process as gp

import sys
from keras.layers.wrappers import TimeDistributed

#Set global variables
dimension_of_hidden_layers = 6
max_depth_glob = 3
number_of_parameters_per_layer_glob = 3
dimension_of_input1 = 2
encoder_decoder = None
encoder_performance = None
no_of_training_data, min_units = None, None
decoder_encoder_M = None
encoder_M = None
decoder_M = None
max_units = None
min_depth = None
num_of_act_fce = None
dimension_of_output_y = 3

def decoder_encoder(decoder, encoder, input):
    layer = decoder(input)
    output = encoder(layer)
    model = Model(inputs=input, outputs = output)
    model.compile(loss='mse', optimizer='adam', metrics=[])
    return model

def local_connected(input1):
    output = LocallyConnected1D(dimension_of_hidden_layers, 1, kernel_initializer='ones', bias_initializer='zeros')(
        input1)
    output = Flatten()(output)
    model = Model(inputs=input1, outputs=output)

    return model

def encoded_decoder(decoder, input1, local):
    layer = local(input1)
    output1, output2 = decoder(layer)
    model = Model(inputs=input1, outputs=[output1, output2])
    model.compile(loss=['mse', "categorical_crossentropy"], optimizer='adam', metrics=[])

    return model


def encoder_model(input1, input2):
    #TimeDistributed Dense layer, for each layer in NN config
    layer = Dense(max_depth_glob*(number_of_parameters_per_layer_glob-1)*2, activation="relu")(input2)

    layer = Dense(max_depth_glob*number_of_parameters_per_layer_glob*2, activation="tanh")(layer)
    layer = Dense(dimension_of_hidden_layers, activation="tanh")(layer)
    #Generate encoded configuration and normalize it
    model = Model(inputs=[input2], outputs=layer)

    return model, layer

def model_for_decoder(input):
    #Repeat the context vector and feed it at every time step
    layer = Dense(max_depth_glob*number_of_parameters_per_layer_glob*2, activation="tanh")(input)
    output1 = Dense(max_depth_glob*3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    output1 = Dense(max_depth_glob*2, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(output1)
    output1 = Dense(max_depth_glob, activation='linear', name="hidden_units")(output1)
    output2 = Dense(max_depth_glob * number_of_parameters_per_layer_glob, activation='relu')(layer)
    output2 = Dense(max_depth_glob * number_of_parameters_per_layer_glob, activation='tanh')(output2)
    output2 = Dense(max_depth_glob*(number_of_parameters_per_layer_glob-1), activation='softmax', activity_regularizer=regularizers.l1(0.1),
        name="act_fce")(output2)
    model = Model(inputs=input,outputs=[ output2])
    model.compile(loss={ "act_fce" : "categorical_crossentropy"}, optimizer='adam', metrics=[])

    return model, output1

def encoder_decoder_construct(input1, input2, encoder, decoder):
    #This builds the whole model together

    layer = encoder(input2)
    output2 = decoder(layer)
    model = Model(inputs=[input2], outputs=[output1, output2])
    model.compile(loss=[ 'mse',  "categorical_crossentropy"], optimizer='adam', metrics=[])

    return model
def encoder_performance_construct(input1, input2, encoder, decoder):

    layer = encoder([ input2])
    output2 = decoder(layer)
    layer = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    layer = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    output3 = Dense(1, activation='linear')(layer)
    model = Model(inputs=[ input2], outputs=[output2,output3])
    model.compile(loss='mse', optimizer='adam', metrics=[])

    return model


def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model2, x_h, x_h_t , x_fce, x_fce_t, epoch, model = None, datax_hidden_perf = None,
                   datax_hidden_t_perf = None, datax_fce_perf = None, datax_fce_t_perf = None,
                   datay_perf = None,batch_size = 10, reverse_order = True):

    len_of_data = x_fce.shape[0]
    len_of_data_perf = None
    if model != None:
        len_of_data_perf = datax_hidden_perf.shape[0]



    if reverse_order:
        x_fce_2 = x_fce_t
        x_h_2 = x_h_t
        x_fce_2_perf = datax_fce_t_perf
        x_h_2_perf = datax_hidden_t_perf
    else:
        x_fce_2 = x_fce
        x_h_2 = x_h
        x_fce_2_perf = datax_fce_perf
        x_h_2_perf = datax_hidden_perf

    no_of_batches = int(len_of_data_perf/ batch_size) + 1

    cur_line, cur_line_perf = 0, 0
    model_loss, model2_loss = [0], []
    rounds_from_last_train_perf = 0
    model_decoder_encoder_loss = []

    for i in range(0,  no_of_batches):

        #Train the performance model
        futur_line_perf = cur_line_perf + batch_size
        if model != None:
            if futur_line_perf <= len_of_data_perf:
                model_loss.append(model.train_on_batch([datax_hidden_perf[cur_line_perf:(futur_line_perf)],
                                                        datax_fce_perf[cur_line_perf:(futur_line_perf)]],
                                                       [x_h_2_perf[cur_line_perf:(futur_line_perf)],
                                                        x_fce_2_perf[cur_line_perf:(futur_line_perf)],
                                                        datay_perf[cur_line_perf:(futur_line_perf)]
                                                        ]))
                cur_line_perf = futur_line_perf
            elif cur_line_perf< len_of_data_perf:
                model_loss.append(model.train_on_batch([datax_hidden_perf[cur_line_perf:(len_of_data_perf)],
                                                        datax_fce_perf[cur_line_perf:(len_of_data_perf)]],
                                                       [x_h_2_perf[cur_line_perf:(len_of_data_perf)],
                                                        x_fce_2_perf[cur_line_perf:(len_of_data_perf)],
                                                        datay_perf[cur_line_perf:(len_of_data_perf)]
                                                        ]))
                cur_line_perf = len_of_data_perf
                rounds_from_last_train_perf = 0


        print("Epoch #{}: model_full Loss: {}".format(epoch + 1, model_loss[-1]))

def train_model(dimension_of_decoder, num_of_act_fce1, min_units1, max_units1, min_depth1, max_depth,
                no_of_training_data1, no_of_parameters_per_layer, dimension_of_output, reverse_order):
    # Callable function from outside to train the model
    # Setting global variables for the models
    global max_depth_glob, number_of_parameters_per_layer_glob, dimension_of_hidden_layers, encoder_decoder, \
        encoder_performance, no_of_training_data, min_units, max_units, min_depth, \
        num_of_act_fce, decoder_encoder_M, decoder_M, encoder_M, dimension_of_output_y
    max_depth_glob = max_depth
    dimension_of_output_y = dimension_of_output
    no_of_training_data, min_units = no_of_training_data1, min_units1
    max_units = max_units1
    min_depth = min_depth1
    num_of_act_fce = num_of_act_fce1

    dimension_of_hidden_layers = dimension_of_decoder
    number_of_parameters_per_layer_glob = no_of_parameters_per_layer

    # Constructing the model
    # Constructing encoder
    input1 = Input(shape=(max_depth*num_of_act_fce,))
    input2 = Input(shape=(max_depth,))
    base_m, base_m_out = encoder_model(input2, input1)
    encoder_M = base_m
    # Constructing decoder
    input = Input(shape=(dimension_of_hidden_layers,))
    decoder, decoder_out = model_for_decoder(input)
    decoder_M = decoder



    # Constructing a encoder_performance model
    input1 = Input(shape=(max_depth*num_of_act_fce,))
    input2 = Input(shape=(max_depth,))
    encoder_performance = encoder_performance_construct(input2, input1, base_m, decoder)

    pkl_file = open('encoder_input3.pkl', 'rb' )

    data1 = pickle.load(pkl_file,  encoding='latin1')
    pkl_file.close()

    pkl_file = open('performance_list.pkl', 'rb')

    data2 = pickle.load(pkl_file)
    pkl_file.close()

    #do data hidenned first
    no_of_training_data_w = data1[0].shape[0]
    data_h  = np.zeros((no_of_training_data_w, max_depth_glob))
    data_fce= np.zeros((no_of_training_data_w, max_depth_glob*(number_of_parameters_per_layer_glob -1)))
    data_h[:,:] = data1[0][:,:,0]

    #do fce
    for i in range(0, no_of_training_data_w):
        for i2 in range(0, max_depth_glob):
            data_fce[i,i2*(number_of_parameters_per_layer_glob-1):(i2+1)*(number_of_parameters_per_layer_glob-1)] = data1[2][i,i2,:]



    datax_hidden,datax_hidden_test, datax_fce, datax_fce_test = data_h[0:1000], data_h[1000:],data_fce[0:1000], data_fce[1000:]

    datax_hidden2, datax_hidden_t2, datax_fce2, datax_fce_t2 = create_first_training_data(10000, min_units,
                                                                                      max_units,
                                                                                      min_depth, max_depth,
                                                                                      num_of_act_fce,
                                                                                      no_of_parameters_per_layer)
    data2 = np.array(data2)[:,1]
    datay_perf = np.array(data2)
    datay_perf_test = datay_perf[1000:]
    datay_perf = datay_perf[:1000]

    for i2 in range(0,1):

        encoder_performance.fit([datax_fce],[ datax_fce, datay_perf], batch_size=10,epochs=100,validation_data=([datax_fce_test],[datax_fce_test,datay_perf_test]))



        encoded_data_test = encoder_M.predict([ datax_fce_test])
        encoded_data = encoder_M.predict([datax_fce])


        for i3 in range(0,20):
            kernel = gp.kernels.Matern(length_scale=(1/(2**i3)))

            alpha = 1e-7
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=20,
                                                normalize_y=True,)
            xp = np.array(encoded_data)
            yp = np.array(data2[:1000])
            model.fit(xp,yp)
            avg = 0
            running_sum_list = []
            for i in range(0, datax_hidden_test.shape[0]):
                to_pred  = np.array(encoded_data_test[i])
                to_pred_2d = np.zeros((1,to_pred.shape[0]))
                to_pred.reshape(1, -1)
                to_pred_2d[0,:] = to_pred
                to_pred = to_pred_2d


                mu, sigma = model.predict(to_pred,return_std=True)
                running_sum = abs(mu - data2[1000 + i])
                avg += running_sum
                running_sum_list.append([running_sum, mu, data2[1000 + i], sigma])
                print(running_sum_list[-1])
            avg = avg / datax_hidden_test.shape[0]
            print(avg, "avg")
            running_sum_list.append([avg, 0, 0, 0])
            running_sum_list = np.array(running_sum_list)
            np.savetxt("error_list{},{}.txt".format(i2,i3), running_sum_list)

def create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth):
    #Creates the bounds for random data which trains the model above
    bounds = np.zeros(((depth)*(1+ num_of_act_fce),2))
    global dimension_of_output_y
    for i in range(0, depth):
        bounds[i*(num_of_act_fce+1),0] = min_units
        bounds[i * (num_of_act_fce + 1), 1] = max_units
        if i == 0:
            bounds[i * (num_of_act_fce + 1), 0] = dimension_of_output_y
            bounds[i * (num_of_act_fce + 1), 1] = dimension_of_output_y
        for j in range(1, num_of_act_fce +1):
            bounds[i * (num_of_act_fce + 1) + j , 0] = 0
            bounds[i * (num_of_act_fce + 1) + j, 1] = 1

    return bounds

def create_first_training_data(no_of_training_data,min_units, max_units,
                               min_depth, max_depth, num_of_act_fce, no_of_parameters_per_layer):
    # Initalize the random data to train upon
    epsilon = 1e-7
    datax_fce = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_fce_t = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_hidden = np.zeros((no_of_training_data, max_depth, 1))
    datax_hidden_t = np.zeros((no_of_training_data, max_depth, 1))
    i = 0
    structures = []
    while i < no_of_training_data:
        depth = int(
            round(np.random.random() * (max_depth - min_depth + 1) * (1 - epsilon) + (min_depth - 0.5 + (epsilon*(max_depth - min_depth + 2 )))))

        bounds = create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth)
        x = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]),
                                         no_of_parameters_per_layer)
        flag = False
        for i2 in range(0, len(structures)):
            if np.array_equal(structures[i2], x):
                flag = True
        if flag:
            continue

        structures.append(x)


        bit_count = 0
        for steps in range(0, max_depth):
            datax_hidden[i, max_depth_glob - steps - 1, 0] = round(x[bit_count])
            bit_count += 1
            for bits_per_layer in range(0, num_of_act_fce):
                datax_fce[i, max_depth_glob - steps - 1, bits_per_layer] = x[bit_count]
                bit_count += 1
        for steps in range(0, max_depth):
            datax_hidden_t[i, steps, :] = datax_hidden[i, max_depth - steps - 1, :]
            datax_fce_t[i, steps, :] = datax_fce[i, max_depth - steps - 1, :]
        i += 1
    return datax_hidden, datax_hidden_t, datax_fce, datax_fce_t

def serialize_next_sample_for_gp(next_sample, number_of_parameters_per_layer):
    #Serializes the random data to trainable form
    global max_depth_glob
    next_sample = next_sample.tolist()
    seriliezed_next_sample = np.zeros((max_depth_glob*number_of_parameters_per_layer))
    number_of_layers = int(len(next_sample) / number_of_parameters_per_layer)

    for i in range(0, number_of_layers):
        #Rounds the number of units in the layer
        seriliezed_next_sample[i * number_of_parameters_per_layer] = \
            round(next_sample[i * number_of_parameters_per_layer])
        #Chooses the activation function
        index = next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer].index(
                max(next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer]))
        for fce in range(1, number_of_parameters_per_layer):
            if index == fce:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce] = 1
            else:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce] = 0
    for i in range(number_of_layers, max_depth_glob):
        for i2 in range(0, number_of_parameters_per_layer):
            seriliezed_next_sample[i* number_of_parameters_per_layer + i2] = 0


    return seriliezed_next_sample



if __name__ == "__main__":
    dimension_of_decoder, num_of_act_fce1, min_units1, max_units1, min_depth1, max_depth,\
    no_of_training_data1, no_of_parameters_per_layer, dimension_of_output, reverse_order = 16, 2, 2, 100, 2, 3, 1000, 3, 3, True
    train_model(dimension_of_decoder, num_of_act_fce1, min_units1, max_units1, min_depth1, max_depth,
                no_of_training_data1, no_of_parameters_per_layer, dimension_of_output, reverse_order)

