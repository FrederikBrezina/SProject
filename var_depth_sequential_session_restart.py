import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import MyCallbacks
import random
import HelpFunctions as hp
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import pickle


data = np.zeros((1000,9))
hash_str_list = []
data_p = [500, 3000]
line = 0
dic = {}
get_bin = lambda x, n: format(x, 'b').zfill(n)
for points in range(0,1):
    data_points = data_p[points]
    dataset = np.loadtxt("new_training_for_simple_reg.txt", delimiter=" ")
    X = dataset[0:data_points, 0:2]
    Y = dataset[0:data_points, 2:5]
    x_test = dataset[-501:-1, 0:2]
    y_test = dataset[-501:-1, 2:5]

    # Types of layers used
    c = ['relu', 'tanh', 'sigmoid']
    c1 = ['01', '10']
    num_of_ouputs = 3

    for nn in range(0,600):
        hash_str = ''
        layers_val = np.zeros((10,2))
        depth = random.randint(2, 10)
        for layer in range(0, depth):
            layers_val[layer][0] = int(98 * random.uniform(0, 1)) + 2
            hash_str += get_bin(int(layers_val[layer][0]), 7)
            layers_val[layer][1] = random.randint(0, 1)

            hash_str += c1[int(layers_val[layer][1])]
        if '{:d}'.format(hash(hash_str)) in dic:
            continue
        dic['{:d}'.format(hash(hash_str))] = hash_str

        #data to monitor
        conv_epoch, max_acc, max_val_acc, diff_of_over_fitting_at_conv, min_val_loss = [], [], [], [], []
        for tries in range(0,4):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            config.gpu_options.allow_growth = True
            ses = tf.InteractiveSession(config=config)
            set_session(ses)
            model = Sequential()

            for layer in range(0, depth):
                if layer == 0:
                    model.add(Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])],
                                    input_shape=(2,)))
                else:
                    model.add(Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])]))

            model.add(Dense(num_of_ouputs, activation='sigmoid'))



            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            acc_his = MyCallbacks.AccHistoryEpochTest()

            model.fit(X, Y, epochs=200, batch_size=16, validation_data=(x_test, y_test),
                      callbacks=[acc_his], shuffle=True)
            #Data Calculation
            conv_epoch.append(hp.convergence_of_NN_val_loss(acc_his.losses_val_losses,4))
            diff_of_over_fitting_at_conv.append(acc_his.losses[conv_epoch[-1]-1] - acc_his.losses_val[conv_epoch[-1]-1])
            max_acc.append(max(acc_his.losses))
            max_val_acc.append(max(acc_his.losses_val))
            min_val_loss.append(min(acc_his.losses_val_losses))
            ses.close()
            tf.reset_default_graph()
        hash_str_list.append(hash_str)
        data[line][0], data[line][1] = hp.mean_and_std(max_acc)
        data[line][2], data[line][3] =   hp.mean_and_std(max_val_acc)
        data[line][4], data[line][5] = hp.mean_and_std(conv_epoch)
        data[line][6], data[line][7] = hp.mean_and_std(diff_of_over_fitting_at_conv)
        data[line][8] = depth
        line += 1
np.savetxt('num_of_layers_var_rand_units_rand_act_for_ally.txt', data, delimiter=" ")
output = open('num_of_layers_var_rand_units_rand_act_for_allx.txt', 'wb')
pickle.dump(hash_str_list, output)
output.close()
