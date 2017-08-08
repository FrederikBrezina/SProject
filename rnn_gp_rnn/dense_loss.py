from keras.models import Sequential
from keras.layers import Dense
import MyCallbacks

def loss_nn_dense(args, x, y, x_test, y_test, act_fce, loss, optimizer, batch_size, c = None, best_last_act = False):
    depth = int(len(args)/2)
    x_dim, y_dim = x.shape[1], y.shape[1]
    model = Sequential()
    for layer in range(0, depth):
        if best_last_act:
            if layer ==0:
                model.add(Dense(int(args[layer*2]), activation=c[int(args[layer*2 + 1])], input_shape = (x_dim,)))
            else:
                model.add(Dense(int(args[layer*2]), activation=c[int(args[layer*2 + 1])]))
        else:
            if layer ==0:
                model.add(Dense(int(args[layer*2]), activation=act_fce[int(args[layer*2 + 1])], input_shape = (x_dim,)))
            else:
                model.add(Dense(int(args[layer*2]), activation=act_fce[int(args[layer*2 + 1])]))
    model.add(Dense(y_dim, activation=act_fce[int(args[-1])]))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    acc_his = MyCallbacks.AccHistoryEpochTest()
    model.fit(x, y, epochs=200, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[acc_his], shuffle=True)

    return min(acc_his.losses_val_losses)

