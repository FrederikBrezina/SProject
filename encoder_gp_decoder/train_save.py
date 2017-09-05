import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import MyCallbacks
import HelpFunctions as hp
from keras.callbacks import EarlyStopping
import pickle
import sys
max_depth_glob = 0
def create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth):
    #Creates the bounds for random data which trains the model above
    bounds = np.zeros(((depth)*(1+ num_of_act_fce),2))
    global dimension_of_output_y
    for i in range(0, depth):
        bounds[i*(num_of_act_fce+1),0] = min_units
        bounds[i * (num_of_act_fce + 1), 1] = max_units
        if i == 0:
            bounds[i * (num_of_act_fce + 1), 0] = 3
            bounds[i * (num_of_act_fce + 1), 1] = 3
        for j in range(1, num_of_act_fce +1):
            bounds[i * (num_of_act_fce + 1) + j , 0] = 0
            bounds[i * (num_of_act_fce + 1) + j, 1] = 1


    return bounds

def serialize_next_sample_for_gp(next_sample, number_of_parameters_per_layer):
    #Serializes the random data to trainable form
    global max_depth_glob

    next_sample = next_sample.tolist()
    seriliezed_next_sample = np.zeros((max_depth_glob*number_of_parameters_per_layer))
    number_of_layers = int(len(next_sample) / number_of_parameters_per_layer)
    to_train_serialized = np.zeros((number_of_layers*2))

    for i in range(0, number_of_layers):
        #Rounds the number of units in the layer
        seriliezed_next_sample[i * number_of_parameters_per_layer] = \
            round(next_sample[i * number_of_parameters_per_layer])
        to_train_serialized[i*2] = round(next_sample[i * number_of_parameters_per_layer])
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


    return seriliezed_next_sample, to_train_serialized

def loss_nn_dense(args, x, y, x_test, y_test, act_fce, loss, optimizer, batch_size):

    depth = int(len(args)/2)

    x_dim, y_dim = x.shape[1], y.shape[1]

    model = Sequential()

    for layer in range(0, depth):
        #Each layer has 2 paramters in args array, number of hidden units, and index of actvation function
        if layer ==0:
            model.add(Dense(int(args[layer*2]), activation=act_fce[int(args[layer*2 + 1])], input_shape = (x_dim,)))
        else:
            model.add(Dense(int(args[layer*2]), activation=act_fce[int(args[layer*2 + 1])]))
    #Last layer
    model.add(Dense(y_dim, activation=act_fce[int(args[-1])]))
    #Loss and optimizer is set by user
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    acc_his = MyCallbacks.AccHistoryEpochTest()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80, verbose=0, mode='min')
    #Batch_size is also set by user
    model.fit(x, y, epochs=300, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[acc_his, early_stopping], shuffle=True)

    conv_epoch = hp.convergence_of_NN_val_loss(acc_his.losses_val_losses, 4)
    if conv_epoch == 0:
        conv_epoch = 1
    diff_of_over_fitting_at_conv = acc_his.losses[conv_epoch - 1] - acc_his.losses_val[conv_epoch - 1]
    max_acc = max(acc_his.losses)
    max_val_acc = max(acc_his.losses_val)
    min_val_loss = min(acc_his.losses_val_losses)

    #Return the best possible test_loss
    return [min_val_loss, max_val_acc, max_acc, diff_of_over_fitting_at_conv, conv_epoch]

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
    structures_to_train = []
    while i < no_of_training_data:
        depth = int(
            round(np.random.random() * (max_depth - min_depth + 1) * (1 - epsilon) + (min_depth - 0.5)))

        bounds = create_bounds(num_of_act_fce, min_units, max_units, depth, max_depth)

        x, x2 = serialize_next_sample_for_gp(np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]),
                                         no_of_parameters_per_layer)
        print(x)

        flag = False
        for i2 in range(0, len(structures)):
            if np.array_equal(structures[i2], x):
                flag = True
        if flag:
            continue

        structures.append(x)
        structures_to_train.append(x2)


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

    return [datax_hidden, datax_hidden_t, datax_fce, datax_fce_t], structures_to_train

if __name__ == "__main__":
    X = np.loadtxt('X_basic_task.txt', delimiter=" ")
    Y = np.loadtxt('Y_basic_task.txt', delimiter=" ")
    test_index = 200
    x, x_test, y, y_test = X[:-test_index], X[-test_index:], Y[:-test_index], Y[-test_index:]

    act_fce = ['relu', 'sigmoid']
    no_of_training_data, min_units, max_units, min_depth,\
    max_depth, num_of_act_fce, no_of_parameters_per_layer = 250, 2, 100, 2, 3, 2, 3
    global max_depth_glob
    max_depth_glob = max_depth

    to_save_encoder, for_dense_nn = create_first_training_data(no_of_training_data,min_units, max_units,
                               min_depth, max_depth, num_of_act_fce, no_of_parameters_per_layer)
    output = open('encoder_input3.pkl', 'wb')
    pickle.dump(to_save_encoder, output)
    output.close()

    performance_metrics = []

    for i in range(0,no_of_training_data):
        cv_score = loss_nn_dense(for_dense_nn[i],x,y,x_test, y_test, act_fce,'categorical_crossentropy', 'adam', 16)
        performance_metrics.append(cv_score)

    output = open('performance_list.pkl', 'wb')
    pickle.dump(performance_metrics, output)
    output.close()

