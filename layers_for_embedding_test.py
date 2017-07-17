import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense
import MyCallbacks
import HelpFunctions as hp
import matplotlib.pyplot as plt
import tensorflow as tf
def main():
    dataset = np.loadtxt("data_for_real1.txt", delimiter=" ")
    X = dataset[0:-101, 0:2]
    Y = dataset[0:-101, 2:5]
    x_test = dataset[-101:-1, 0:2]
    y_test = dataset[-101:-1, 2:5]

    epochs = 30
    batch_size = 20
    # Types of layers used
    c = ['relu', 'tanh', 'sigmoid']
    num_of_ouputs = 3
    input = Input(shape=(2,))
    layers = Dense(100, activation=c[0])(input)
    layers = Dense(40, activation=c[1])(layers)
    layers = Dense(30, activation=c[1])(layers)
    layers = Dense(20, activation=c[2])(layers)
    output = Dense(num_of_ouputs, activation='sigmoid')(layers)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    acc_his = MyCallbacks.AccHistoryEpochTest()
    loss_his_epoch = MyCallbacks.LossHistoryEpoch()
    val_loss_his = MyCallbacks.ValLossHistory()
    loss_his = MyCallbacks.ValLossHistory()

    model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[acc_his, val_loss_his,loss_his_epoch,  loss_his], shuffle=True)
    f = [4, 3, 2, 1, 0]
    g = f[0:4]
    h = f[1:5]

    k = tf.subtract(g, h) + tf.subtract(h, g)
    ses = tf.Session()
    print(ses.run(k))
    print(hp.overfitting_all_values(loss_his_epoch.losses, val_loss_his.losses, hp.convergence_of_NN_val_loss(loss_his.losses, 4)))
    print(len(model.layers))

    print(hp.convergence_of_NN_val_loss(loss_his.losses, 4))

    plt.plot(acc_his.losses, 'b-', acc_his.losses_val, 'r-')
    plt.show()

if __name__ == '__main__':
    main()