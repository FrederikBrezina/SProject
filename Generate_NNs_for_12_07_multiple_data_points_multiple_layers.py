import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import random

data = np.zeros((1000,3))
data_p = [500, 3000]
line = 0
dic = {}
get_bin = lambda x, n: format(x, 'b').zfill(n)
for points in range(1,2):
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

    for nn in range(0,1000):
        hash_str = ''
        input = Input(shape=(2,))
        for layer in range(0,3):
            act_units1 = int(100*random.uniform(0,1))
            hash_str += get_bin(act_units1, 7)
            act_fce1 = random.randint(0,1)
            hash_str += c1[act_fce1]
            if layer ==0 :
                layers = Dense(act_units1, activation=c[act_fce1])(input)
            else:
                layers = Dense(act_units1, activation=c[act_fce1])(layers)
        if '{:d}'.format(hash(hash_str)) in dic:
            continue
        print(hash_str)
        dic['{:d}'.format(hash(hash_str))] = hash_str
        output = Dense(num_of_ouputs, activation='sigmoid')(layers)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        acc_his = MyCallbacks.AccHistoryEpochTest()

        model.fit(X, Y, epochs=2, batch_size=15, validation_data=(x_test, y_test),
                  callbacks=[acc_his], shuffle=True)
        data[line][0], data[line][1], data[line][2] = int(hash_str, base=2),  max(acc_his.losses), max(acc_his.losses_val)
        line += 1
np.savetxt('num_of_layers_3_rand_units_rand_act_for_all.txt', data, delimiter=" ")
