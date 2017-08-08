from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, concatenate
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np

from keras.layers.wrappers import TimeDistributed

#Set global variables
dimension_of_hidden_layers = 0
max_depth_glob = 0
number_of_parameters_per_layer_glob = 0
dimension_of_input1 = 2

def encoder_model(input1, input2):
    #TimeDistributed Dense layer, for each layer in NN config
    layer = TimeDistributed(Dense(dimension_of_input1))(input1)
    layer2 = TimeDistributed(Dense(dimension_of_hidden_layers))(input2)
    layer = concatenate([layer, layer2])
    #Apply the LSTM to each layer which passed thourgh dense first
    layer = LSTM(dimension_of_hidden_layers, kernel_regularizer=regularizers.l2(0.01),
                 dropout=0.05, return_sequences=False)(layer)
    #Generate encoded configuration and normalize it
    layer = BatchNormalization()(layer)
    model = Model(inputs=[input1, input2], outputs=layer)


    return model, layer

def model_for_decoder(input):
    #Repeat the context vector and feed it at every time step
    layer = RepeatVector(max_depth_glob)(input) # Get the last output of the GRU and repeats it
    #Return the sequence into time distributed dense network
    output = LSTM(dimension_of_hidden_layers,  kernel_regularizer=regularizers.l2(0.01),
                   dropout=0.05, return_sequences=True, name='lstm_output')(layer)
    #Last layer, Dense layer before the output prediction and reconstruction of the input
    output1 = TimeDistributed(Dense(1), name="hidden_units")(output)
    output2 = TimeDistributed(Dense(number_of_parameters_per_layer_glob - 1), name="act_fce")(output)
    model = Model(inputs=input,outputs=[output1, output2])
    model.compile(loss={"hidden_units" : 'mse', "act_fce" : "categorical_crossentropy"}, optimizer='adam', metrics=[])

    return model, output1

def encoder_decoder_construct(input1, input2, encoder, decoder):
    #This builds the whole model together

    layer = encoder([input1, input2])
    output1, output2 = decoder(layer)
    model = Model(inputs=[input1,input2], outputs=[output1, output2])
    model.compile(loss=[ 'mse',  "categorical_crossentropy"], optimizer='adam', metrics=[])

    return model

def set_trainable(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train_on_epoch(model2, x_fce, x_fce_t, x_h, x_h_t , epoch, batch_size = 10, reverse_order = True):
    len_of_data = x_fce.shape[0]
    cur_line = 0
    model_loss, model2_loss = [], []
    if reverse_order:
        x_fce_2 = x_fce_t
        x_h_2 = x_h_t
    else:
        x_fce_2 = x_fce
        x_h_2 = x_h
    for i in range(0, int(len_of_data/ batch_size) + 1):
        futur_line = cur_line + batch_size
        if (futur_line)<= len_of_data:
            model2_loss.append(model2.train_on_batch([x_h[cur_line:(futur_line)], x_fce[cur_line:(futur_line)]],
                                                     [x_h_2[cur_line:(futur_line)], x_fce_2[cur_line:(futur_line)]
                                                      ]))
            cur_line = futur_line
        elif (cur_line<len_of_data):
            model2_loss.append(model2.train_on_batch([x_h[cur_line:(len_of_data)], x_fce[cur_line:(len_of_data)]],
                                                     [x_h_2[cur_line:(len_of_data)], x_fce_2[cur_line:(len_of_data)]]))

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
    input1 = Input(shape=(max_depth, num_of_act_fce,))
    input2 = Input(shape=(max_depth, 1,))
    base_m, base_m_out = encoder_model(input2, input1)
    #Constructing decoder
    input = Input(shape=(dimension_of_hidden_layers,))
    decoder, decoder_out = model_for_decoder(input)
    #Constructing the whole model
    input1 = Input(shape=(max_depth, num_of_act_fce,))
    input2 = Input(shape=(max_depth, 1,))
    full_model= encoder_decoder_construct(input2, input1,base_m,decoder)

    #Initalize the random data to train upon
    datax_fce = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_fce_t = np.zeros((no_of_training_data, max_depth, num_of_act_fce))
    datax_hidden = np.zeros((no_of_training_data, max_depth, 1))
    datax_hidden_t = np.zeros((no_of_training_data, max_depth, 1))

    for i in range(0, no_of_training_data):
        depth = int(round(np.random.random() * (max_depth - min_depth + 1) * (1 - epsilon) + (min_depth - 0.5 + epsilon)))
        bounds = create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth)
        x = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]), no_of_parameters_per_layer)
        bit_count = 0
        for steps in range(0, max_depth):
            datax_hidden[i, steps, 0] = x[bit_count]
            bit_count += 1
            for bits_per_layer in range(0, num_of_act_fce):
                datax_fce[i, steps, bits_per_layer] = x[bit_count]
                bit_count += 1
        for steps in range(0, max_depth):
            datax_hidden_t[i, steps, :] = datax_hidden[i, max_depth - steps - 1, :]
            datax_fce_t[i,steps,:] = datax_fce[i,max_depth - steps -1,:]

    #Train the encoder_decoder
    for epoch in range(0,150):
        train_on_epoch(full_model, datax_fce, datax_fce_t, datax_hidden, datax_hidden_t, epoch, 10, reverse_order=reverse_order)
    #Return encoder and decoder
    return base_m, decoder


