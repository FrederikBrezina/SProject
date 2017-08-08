from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np

from keras.layers.wrappers import TimeDistributed

#Set global variables
dimension_of_hidden_layers = 0
max_depth_glob = 0
number_of_parameters_per_layer_glob = 0

def encoder_model(input):
    #TimeDistributed Dense layer, for each layer in NN config
    layer = TimeDistributed(Dense(dimension_of_hidden_layers))(input)
    #Apply the LSTM to each layer which passed thourgh dense first
    layer = LSTM(dimension_of_hidden_layers, kernel_regularizer=regularizers.l2(0.01),
                 dropout=0.05, return_sequences=False)(layer)
    #Generate encoded configuration and normalize it
    layer = BatchNormalization()(layer)
    model = Model(inputs=input, outputs=layer)
    model.compile(loss='mse', optimizer='adam', metrics=[])

    return model, layer

def model_for_decoder(input):
    #Repeat the context vector and feed it at every time step
    layer = RepeatVector(max_depth_glob)(input) # Get the last output of the GRU and repeats it
    #Return the sequence into time distributed dense network
    output1 = LSTM(dimension_of_hidden_layers,  kernel_regularizer=regularizers.l2(0.01),
                   dropout=0.05, return_sequences=True, name='lstm_output')(layer)
    #Last layer, Dense layer before the output prediction and reconstruction of the input
    output1 = TimeDistributed(Dense(number_of_parameters_per_layer_glob))(output1)
    model = Model(inputs=input,outputs=output1)
    model.compile(loss='mse', optimizer='adam', metrics=[])

    return model, output1

def encoder_decoder_construct(input, encoder, decoder):
    #This builds the whole model together
    layer = encoder(input)
    output = decoder(layer)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=[])

    return model, output

def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model2, x, x_t, epoch, batch_size = 10, reverse_order = True):
    len_of_data = x.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    if reverse_order:
        x_2 = x_t
    else:
        x_2 = x
    for i in range(0, int(len_of_data/ batch_size) + 1):
        futur_line = cur_line + batch_size
        if (futur_line)<= len_of_data:
            model2_loss.append(model2.train_on_batch(x[cur_line:(futur_line)], x_2[cur_line:(futur_line)]))
            cur_line = futur_line
        elif (cur_line<len_of_data):
            model2_loss.append(model2.train_on_batch(x[cur_line:len_of_data], x_2[cur_line:len_of_data]))

        print("Epoch #{}: model Loss: {}".format(epoch + 1,model2_loss[-1]))


def create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth):
    #Creates the bounds for random data which trains the model above
    bounds = np.zeros(((max_depth)*(1+ num_of_act_fce),2))

    for i in range(max_depth - depth, max_depth):
        bounds[i*(num_of_act_fce+1),0] = min_units
        bounds[i * (num_of_act_fce + 1), 1] = max_units
        for j in range(1, num_of_act_fce +1):
            bounds[i * (num_of_act_fce + 1) + j , 0] = 0
            bounds[i * (num_of_act_fce + 1) + j, 1] = 1

    return bounds

def serialize_next_sample_for_gp(next_sample, number_of_parameters_per_layer):
    #Serializes the random data to trainable form
    next_sample = next_sample.tolist()
    seriliezed_next_sample = np.copy(next_sample)
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

    return seriliezed_next_sample

def train_model(dimension_of_decoder, num_of_act_fce, min_units, max_units,min_depth, max_depth,
                no_of_training_data, no_of_parameters_per_layer, reverse_order):
    #Callable function from outside to train the model
    epsilon = 1e-7

    #Setting global variables for the models
    global max_depth_glob
    max_depth_glob = max_depth
    global number_of_parameters_per_layer_glob
    global dimension_of_hidden_layers
    dimension_of_hidden_layers = dimension_of_decoder
    number_of_parameters_per_layer_glob = no_of_parameters_per_layer

    #Constructing the model
    #Constructing encoder
    input = Input(shape=(max_depth, no_of_parameters_per_layer,))
    base_m, base_m_out = encoder_model(input)
    #Constructing decoder
    input = Input(shape=(dimension_of_hidden_layers,))
    decoder, decoder_out = model_for_decoder(input)
    #Constructing the whole model
    input = Input(shape=(max_depth, no_of_parameters_per_layer,))
    full_model, full_model_out = encoder_decoder_construct(input,base_m,decoder)

    #Initalize the random data to train upon
    datax = np.zeros((no_of_training_data, max_depth, no_of_parameters_per_layer))
    datax_t = np.zeros((no_of_training_data, max_depth, no_of_parameters_per_layer))
    for i in range(0, no_of_training_data):
        depth = int(round(np.random.random() * (max_depth - min_depth + 1) * (1 - epsilon) + (min_depth - 0.5 + epsilon)))
        bounds = create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth)
        x = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]), no_of_parameters_per_layer)
        bit_count = 0
        for steps in range(0, max_depth):
            for bits_per_layer in range(0, no_of_parameters_per_layer):
                datax[i, steps, bits_per_layer] = x[bit_count]
                bit_count += 1
        for steps in range(0, max_depth):
            datax_t[i,steps,:] = datax[i,max_depth - steps -1,:]

    #Train the encoder_decoder
    for epoch in range(0,150):
        train_on_epoch(full_model, datax, datax_t, epoch, 10, reverse_order=reverse_order)
    #Return encoder and decoder
    return base_m, decoder


