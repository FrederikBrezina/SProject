from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, concatenate, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers.local import LocallyConnected1D
import numpy as np
import sys
from encoder_gp_decoder.dense_loss import loss_nn_dense

import pickle


from keras.layers.wrappers import TimeDistributed

#Set global variables
dimension_of_hidden_layers = 0
max_depth_glob = 0
number_of_parameters_per_layer_glob = 0
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
dimension_of_output_y = 0
dimension_of_encoder = 0


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
    layer = TimeDistributed(Dense(dimension_of_input1,  kernel_regularizer=regularizers.l2(0.005), activation='relu'))(input1)
    layer2 = TimeDistributed(Dense(dimension_of_encoder, kernel_regularizer=regularizers.l2(0.005), activation='relu'))(input2)
    layer = concatenate([layer, layer2])
    #Apply the LSTM to each layer which passed thourgh dense first
    layer = LSTM(dimension_of_encoder, kernel_regularizer=regularizers.l2(0.01),
                 return_sequences=True, activation='tanh')(layer)
    layer = LSTM(dimension_of_encoder, kernel_regularizer=regularizers.l2(0.01),
                 return_sequences=False, activation='tanh')(layer)
    layer = Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    layer = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    layer = Dense(dimension_of_hidden_layers, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(layer)
    #Generate encoded configuration and normalize it
    model = Model(inputs=[input1, input2], outputs=layer)
    model.compile(loss='mse', optimizer='adam', metrics=[])

    return model, layer

def model_for_decoder(input):
    #Repeat the context vector and feed it at every time step
    layer = RepeatVector(max_depth_glob)(input) # Get the last output of the GRU and repeats it
    #Return the sequence into time distributed dense network
    output = LSTM(dimension_of_hidden_layers,  kernel_regularizer=regularizers.l2(0.005),
                   return_sequences=True, name='lstm_output')(layer)
    #Last layer, Dense layer before the output prediction and reconstruction of the input
    output1 = TimeDistributed(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.005)))(output)
    output1 = TimeDistributed(Dense(1, activation='relu'), name="hidden_units")(output1)
    output2 = TimeDistributed(Dense(number_of_parameters_per_layer_glob - 1, activation='softmax', activity_regularizer=regularizers.l1(0.1)), name="act_fce")(output)
    model = Model(inputs=input,outputs=[output1, output2])
    model.compile(loss={"hidden_units" : 'mse', "act_fce" : "categorical_crossentropy"}, optimizer='adam', metrics=[])

    return model, output1

def encoder_decoder_construct(input1, input2, encoder, decoder):
    #This builds the whole model together

    layer = encoder([input1, input2])
    output1, output2 = decoder(layer)
    model = Model(inputs=[input1,input2], outputs=[output1, output2])
    model.compile(loss="mse", optimizer='adam', metrics=[],loss_weights=[0.1,400.])

    return model
def encoder_performance_construct(input1, input2, encoder, decoder):

    layer = encoder([input1, input2])
    output1, output2 = decoder(layer)
    layer = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    layer = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    output3 = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=[input1, input2], outputs=[output1,output2,output3])
    model.compile(loss="mse", optimizer='adam', metrics=[], loss_weights=[0.1,1000.,10000.])

    return model


def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model2, x_h, x_h_t , x_fce, x_fce_t, epoch, model = None, datax_hidden_perf = None,
                   datax_hidden_t_perf = None, datax_fce_perf = None, datax_fce_t_perf = None,
                   datay_perf = None,batch_size = 11, reverse_order = True):

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

    no_of_batches = int(len_of_data/ batch_size) + 1

    cur_line, cur_line_perf = 0, 0
    model_loss, model2_loss = [0], [0]
    rounds_from_last_train_perf = 0
    model_decoder_encoder_loss = []

    for i in range(0,  no_of_batches):
        futur_line = cur_line + batch_size
        if (futur_line)<= len_of_data:
            model2_loss.append(model2.train_on_batch([x_h[cur_line:(futur_line)], x_fce[cur_line:(futur_line)]],
                                                     [x_h_2[cur_line:(futur_line)], x_fce_2[cur_line:(futur_line)]
                                                      ]))
            #Train only encoder
            cur_line = futur_line
        elif (cur_line<len_of_data):

            model2_loss.append(model2.train_on_batch([x_h[cur_line:(len_of_data)], x_fce[cur_line:(len_of_data)]],
                                                     [x_h_2[cur_line:(len_of_data)], x_fce_2[cur_line:(len_of_data)]]))

        #Train the performance model
        futur_line_perf = cur_line_perf + batch_size
        if model != None:
            if futur_line <= len_of_data_perf:
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
                cur_line_perf = 0
            else:
                cur_line_perf = 0

        print("Epoch #{}: model_full Loss: {}, model_perf_Loss: {}".format(epoch + 1,model2_loss[-1], model_loss[-1]))


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

def serialize_next_sample_for_gp(next_sample, number_of_parameters_per_layer,train):
    #Serializes the random data to trainable form
    global max_depth_glob
    next_sample = next_sample.tolist()
    seriliezed_next_sample = np.zeros((max_depth_glob*number_of_parameters_per_layer))
    number_of_layers = int(len(next_sample) / number_of_parameters_per_layer)
    to_train_serialized = np.zeros((number_of_layers * 2))


    for i in range(0, number_of_layers):
        #Rounds the number of units in the layer
        seriliezed_next_sample[i * number_of_parameters_per_layer] = \
            round(next_sample[i * number_of_parameters_per_layer])
        to_train_serialized[i * 2] = round(next_sample[i * number_of_parameters_per_layer])
        #Chooses the activation function
        index = next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer].index(
                max(next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer]))
        to_train_serialized[i * 2 + 1] = index
        for fce in range(0, number_of_parameters_per_layer-1):
            if index == fce:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce + 1] = 1
            else:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce + 1] = 0
    for i in range(number_of_layers, max_depth_glob):
        for i2 in range(0, number_of_parameters_per_layer):
            seriliezed_next_sample[i* number_of_parameters_per_layer + i2] = 0
    sanitized_example = seriliezed_next_sample[:len(next_sample)]
    if train:
        return seriliezed_next_sample, to_train_serialized, sanitized_example
    else:
        return seriliezed_next_sample

def create_first_training_data(no_of_training_data,min_units, max_units,
                               min_depth, max_depth, num_of_act_fce, no_of_parameters_per_layer, train):
    # Initalize the random data to train upon
    epsilon = 1e-7
    datax_fce = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_fce_t = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_hidden = np.zeros((no_of_training_data, max_depth, 1))
    datax_hidden_t = np.zeros((no_of_training_data, max_depth, 1))
    i = 0
    structures = []
    structures_to_train = []
    sanitized_example_list= []
    while i < no_of_training_data:
        depth = int(
            round(np.random.random() * (max_depth - min_depth + 1) * (1 - epsilon) + (min_depth - 0.5 + (epsilon*(max_depth - min_depth + 2 )))))

        bounds = create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth)
        if train:
            x, x2, sanitized_example = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]),
                                             no_of_parameters_per_layer, train)

        else:
            x = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]),
                                         no_of_parameters_per_layer, train)
        flag = False
        for i2 in range(0, len(structures)):
            if np.array_equal(structures[i2], x):
                flag = True
        if flag:
            continue

        structures.append(x)
        if train:
            structures_to_train.append(x2)
            sanitized_example_list.append(sanitized_example)


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
    if train:
        return datax_hidden, datax_hidden_t, datax_fce, datax_fce_t, structures_to_train, sanitized_example_list
    else:
        return datax_hidden, datax_hidden_t, datax_fce, datax_fce_t

def transform_into_timeseries(datax):

    max_depth = max_depth_glob
    num_of_act_fce = number_of_parameters_per_layer_glob - 1

    length_of_datax = len(datax)
    datax_hidden_perf, datax_hidden_t_perf = np.zeros((length_of_datax, max_depth_glob, 1)), \
                                             np.zeros((length_of_datax, max_depth_glob, 1))

    datax_fce_perf, datax_fce_t_perf = np.zeros(
        (length_of_datax, max_depth_glob, number_of_parameters_per_layer_glob - 1)), np.zeros(
        (length_of_datax, max_depth_glob, number_of_parameters_per_layer_glob - 1))



    for i in range(0, length_of_datax):
        bit_count = 0
        act_len_of_datax = len(datax[i])
        steps = 0

        while bit_count < act_len_of_datax:
            datax_hidden_t_perf[i, steps, 0] = datax[i][bit_count]
            bit_count += 1
            try:
                for bits_per_layer in range(0, num_of_act_fce):
                    datax_fce_t_perf[i, steps, bits_per_layer] = datax[i][bit_count]

                    bit_count += 1
            except (IndexError):
                pass
            steps += 1

        #Transpose it, for reverse order
        for steps2 in range(0, max_depth_glob):
            datax_hidden_perf[i, steps2, :] = datax_hidden_t_perf[i, max_depth - steps2 - 1, :]
            datax_fce_perf[i, steps2, :] = datax_fce_t_perf[i, max_depth - steps2 - 1, :]



        return datax_hidden_perf, datax_hidden_t_perf, datax_fce_perf, datax_fce_t_perf


def train_model(x, y, x_test, y_test, act_fce,loss, optimizer, dimension_of_decoder, num_of_act_fce1, min_units1, max_units1,min_depth1, max_depth,
                no_of_training_data1, no_of_parameters_per_layer,dimension_of_output, reverse_order, initial_search = 100):
    from encoder_gp_decoder.gp_bayes_encoder import sanitize_next_sample_for_gp, seriliaze_next_sample_for_loss_fce
    #Callable function from outside to train the model
    #Setting global variables for the models
    global max_depth_glob, number_of_parameters_per_layer_glob, dimension_of_hidden_layers, encoder_decoder, \
        encoder_performance, no_of_training_data, min_units, max_units, min_depth, \
        num_of_act_fce, decoder_encoder_M, decoder_M, encoder_M, dimension_of_output_y, dimension_of_encoder
    max_depth_glob = max_depth
    dimension_of_output_y = dimension_of_output
    no_of_training_data, min_units = no_of_training_data1, min_units1
    max_units = max_units1
    min_depth = min_depth1
    num_of_act_fce = num_of_act_fce1




    number_of_parameters_per_layer_glob = no_of_parameters_per_layer
    dimension_of_hidden_layers = dimension_of_decoder
    dimension_of_encoder =  max_depth_glob * number_of_parameters_per_layer_glob * 2
    #Constructing the model
    #Constructing encoder
    input1 = Input(shape=(max_depth, num_of_act_fce,))
    input2 = Input(shape=(max_depth, 1,))
    base_m, base_m_out = encoder_model(input2, input1)
    encoder_M = base_m
    #Constructing decoder
    input = Input(shape=(dimension_of_hidden_layers,))
    decoder, decoder_out = model_for_decoder(input)
    decoder_M = decoder
    #Constructing the whole model encoder_decoder
    input1 = Input(shape=(max_depth, num_of_act_fce,))
    input2 = Input(shape=(max_depth, 1,))
    full_model= encoder_decoder_construct(input2, input1,base_m,decoder)
    encoder_decoder = full_model
    #Construct the decoder_encoder
    input = Input(shape=(dimension_of_hidden_layers,))
    decoder_encoder_M = decoder_encoder(decoder,base_m, input)



    #Constructing a encoder_performance model
    input1 = Input(shape=(max_depth, num_of_act_fce,))
    input2 = Input(shape=(max_depth, 1,))
    encoder_performance = encoder_performance_construct(input2, input1, base_m, decoder)

    datax_hidden, datax_hidden_t, datax_fce, datax_fce_t = create_first_training_data(no_of_training_data, min_units,
                                                                                      max_units,
                                                                                      min_depth, max_depth,
                                                                                      num_of_act_fce,
                                                                                      no_of_parameters_per_layer, train=False)

    datax_hidden2, datax_hidden_t2, datax_fce2, datax_fce_t2, for_dense_nn, sanitized_list = create_first_training_data(initial_search, min_units,
                                                                                          max_units,
                                                                                          min_depth, max_depth,
                                                                                          num_of_act_fce,
                                                                                          no_of_parameters_per_layer, train=True)
    ############################################################################################
    pkl_file = open('encoder_input3.pkl', 'rb')

    data1 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('performance_list.pkl', 'rb')

    data2 = pickle.load(pkl_file)
    pkl_file.close()

    number_of_datac = initial_search
    datax_hidden2, datax_hidden_t2, \
    datax_fce2, datax_fce_t2 = data1[0][0:number_of_datac], data1[1][0:number_of_datac], data1[2][0:number_of_datac], data1[3][0:number_of_datac]

    sanitized_list, for_dense_nn = [],[]
    for i in range(number_of_datac):

        sanitized_list.append(sanitize_next_sample_for_gp([datax_hidden_t2[i:i+1,:,:], datax_fce_t2[i:i+1,:,:]],no_of_parameters_per_layer,min_units,max_units,dimension_of_output_y))
        for_dense_nn.append(seriliaze_next_sample_for_loss_fce(sanitized_list[-1],number_of_parameters_per_layer_glob))

    #############################################################################################
    # performance_metrics = []
    # for i in range(0,len(for_dense_nn)):
    #     cv_score = loss_nn_dense(for_dense_nn[i],x,y,x_test, y_test, act_fce,loss, optimizer, 16)
    #     performance_metrics.append(cv_score)
    #     assert len(performance_metrics) == i+1
    #     print("NN_in_intial search_____: ", i+1)
    #
    # performance_metrics_arry = np.array(performance_metrics)
    #################################################
    performance_metrics = []
    performance_metrics_arry = np.array(data2)[:number_of_datac,:]
    for i in range(performance_metrics_arry.shape[0]):
        performance_metrics.append(performance_metrics_arry[i,:])
    ####################################################
    #Train the encoder_decoder
    for epoch in range(0,200):
        train_on_epoch(full_model, datax_hidden, datax_hidden_t, datax_fce, datax_fce_t, epoch,encoder_performance,
                       datax_hidden2,datax_hidden_t2,datax_fce2,datax_fce_t2, performance_metrics_arry[:,1], batch_size=10,
                       reverse_order=reverse_order)

    #Return encoder and decoder, full model
    return base_m, decoder, full_model, [datax_hidden2, datax_hidden_t2,
                                         datax_fce2, datax_fce_t2, sanitized_list, for_dense_nn, performance_metrics]

def train_all_models(datax, datay):
    datax_hidden_perf, datax_hidden_t_perf, datax_fce_perf, datax_fce_t_perf = transform_into_timeseries(datax)

    #Do datay separately
    length_of_datax = len(datax)
    assert length_of_datax == len(datay)
    datay_perf = np.zeros((length_of_datax, 1))
    for i in range(0,length_of_datax):
        # Do datay now
        datay_perf[i, :] = datay[i][1]

    datax_hidden, datax_hidden_t, datax_fce, datax_fce_t = create_first_training_data(no_of_training_data, min_units,
                                                                                      max_units,
                                                                                      min_depth, max_depth_glob,
                                                                                      num_of_act_fce,
                                                                                      number_of_parameters_per_layer_glob, train=False)

    for epoch in range(0, 40):
        train_on_epoch(encoder_decoder, datax_hidden, datax_hidden_t, datax_fce, datax_fce_t, epoch,
                       encoder_performance, datax_hidden_perf,
                       datax_hidden_t_perf, datax_fce_perf, datax_fce_t_perf,
                       datay_perf, batch_size=10, reverse_order=True)

def predict_encoded(output):
    return encoder_M.predict(output)

