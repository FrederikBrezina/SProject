from keras.models import Sequential
from keras.layers import Dense
import MyCallbacks

#Modifiable dense NN_architecture
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
    #Batch_size is also set by user
    model.fit(x, y, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[acc_his], shuffle=True)

    #Return the best possible test_loss
    return [min(acc_his.losses_val_losses),0]

