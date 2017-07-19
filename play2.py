import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import random
import HelpFunctions as hp
#import multi_gpu as mg


data = np.zeros((1000,9))
data_p = [500, 3000]
line = 0
dic = {}
get_bin = lambda x, n: format(x, 'b').zfill(n)
std_tries = 4
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

    for nn in range(0,350):
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

        #data to monitor
        conv_epoch, max_acc, max_val_acc, diff_of_over_fitting_at_conv = [], [], [], []
        inputs, outputs = [], []
        for tries in range(0,std_tries):
            name = 'test_input{0:d}'.format(tries)
            input = Input(shape=(2,), name=name)
            inputs.append(input)
            for layer in range(0,3):
                if layer ==0 :
                    layers = Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])])(input)
                else:
                    layers = Dense(int(layers_val[layer][0]), activation=c[int(layers_val[layer][1])])(layers)

            name = 'output{0:d}'.format(tries)
            output = Dense(num_of_ouputs, activation='sigmoid', name=name)(layers)
            outputs.append(output)
        model = Model(inputs=inputs, outputs=outputs)
        #model = mg.make_parallel(model, ['/gpu:0', '/gpu:1'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        acc_his = MyCallbacks.MultiModelAcc(4)
        loss_his = MyCallbacks.MultiModelLosses(4)

        model.fit([X,X,X,X], [Y,Y,Y,Y], epochs=200, batch_size=20, validation_data=([x_test,x_test,x_test,x_test], [y_test,y_test,y_test,y_test]),
                  callbacks=[acc_his, loss_his], shuffle=True)
        #Data Calculation
        for num_of_output in range(0,std_tries):
            conv_epoch.append(hp.convergence_of_NN_val_loss(loss_his.losses[(num_of_output*2) + 1],4))
            diff_of_over_fitting_at_conv.append(acc_his.acc[(num_of_output*2)][conv_epoch[-1]-1] - acc_his.acc[(num_of_output*2) + 1][conv_epoch[-1]-1])
            max_acc.append(max(acc_his.acc[(num_of_output*2)]))
            max_val_acc.append(max(acc_his.acc[(num_of_output*2) + 1]))
        data[line][0] =int(hash_str, base=2)
        data[line][1], data[line][2] = hp.mean_and_std(max_acc)
        data[line][3], data[line][4] =   hp.mean_and_std(max_val_acc)
        data[line][5], data[line][6] = hp.mean_and_std(conv_epoch)
        data[line][7], data[line][8] = hp.mean_and_std(diff_of_over_fitting_at_conv)
        line += 1
np.savetxt('num_of_layers_3_rand_units_rand_act_for_all.txt', data, delimiter=" ")



