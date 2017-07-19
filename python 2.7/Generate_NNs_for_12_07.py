import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import random
import HelpFunctions as hp

data = np.zeros((500,9))
data_p = [500, 3000]
line = 0
dic = {}
get_bin = lambda x, n: format(x, 'b').zfill(n)
for points in range(0,1):
    data_points = data_p[points]
    dataset = np.loadtxt("new_training_for_simple_reg.txt", delimiter=" ")
    X = dataset[0:data_points, 0:2]
    Y = dataset[0:data_points, 2:5]
    x_test = dataset[-10000:-1, 0:2]
    y_test = dataset[-10000:-1, 2:5]

    # Types of layers used
    c = ['relu', 'tanh', 'sigmoid']
    c1 = ['01', '10']
    num_of_ouputs = 3

    for nn in range(0,2):
        hash_str = ''
        layers_val = np.zeros((3,2))
        for layer in range(0, 3):
            layers_val[layer][0] = int(100 * random.uniform(0, 1))
            hash_str += get_bin(int(layers_val[layer][0]), 7)
            layers_val[layer][1] = random.randint(0, 1)

            hash_str += c1[int(layers_val[layer][1])]
        if '{:d}'.format(hash(hash_str)) in dic:
            continue
        dic['{:d}'.format(hash(hash_str))] = hash_str
        print nn/5
        #data to monitor
        conv_epoch, max_acc, max_val_acc, diff_of_over_fitting_at_conv = [], [], [], []
        for tries in range(0,5):
            input = Input(shape=(2,))
            for layer in range(0,3):
                if layer ==0 :
                    layers = Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])])(input)
                else:
                    layers = Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])])(layers)

            output = Dense(num_of_ouputs, activation='sigmoid')(layers)
            model = Model(inputs=input, outputs=output)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            acc_his = MyCallbacks.AccHistoryEpochTest()

            model.fit(X, Y, epochs=200, batch_size=20, validation_data=(x_test, y_test),
                      callbacks=[acc_his], shuffle=True)
            #Data Calculation
            conv_epoch.append(hp.convergence_of_NN_val_loss(acc_his.losses_val_losses,4))
            diff_of_over_fitting_at_conv.append(acc_his.losses[conv_epoch[-1]-1] - acc_his.losses[conv_epoch[-1]-1])
            max_acc.append(max(acc_his.losses))
            max_val_acc.append(max(acc_his.losses_val))
        data[line][0] =int(hash_str, base=2)
        data[line][1], data[line][2] = hp.mean_and_std(max_acc)
        data[line][3], data[line][4] =   hp.mean_and_std(max_val_acc)
        data[line][5], data[line][6] = hp.mean_and_std(conv_epoch)
        data[line][7], data[line][8] = hp.mean_and_std(diff_of_over_fitting_at_conv)
        line += 1
np.savetxt('num_of_layers_3_rand_units_rand_act_for_all.txt', data, delimiter=" ")
