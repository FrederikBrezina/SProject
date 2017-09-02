from keras.models import Sequential
from keras.layers import Dense
import MyCallbacks
import HelpFunctions as hp
from keras.callbacks import EarlyStopping

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
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=90, verbose=0, mode='min')
    #Batch_size is also set by user
    model.fit(x, y, epochs=320, batch_size=batch_size, validation_data=(x_test, y_test),
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

