import pickle
import numpy as np
from rnn_gp_rnn.gp_bayes_encoder import bayesian_optimisation
def load_data(dataxname, datayname):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    def shuffle(a, b):
        perm = np.random.permutation(b.shape[0])
        return a[perm], b[perm]
    ####load data
    pkl_file = open(dataxname, 'rb')
    datax = pickle.load(pkl_file)
    datay = np.loadtxt(datayname, delimiter=" ")
    time_distribution_steps = 10
    number_of_bits_per_layer = 9
    number_of_data = len(datax)
    X = np.zeros((number_of_data, time_distribution_steps, number_of_bits_per_layer))
    for i in range(0, number_of_data):
        bit = get_bin(int(datax[i], base=2), time_distribution_steps * number_of_bits_per_layer)
        bit_count = 0
        for steps in range(0, time_distribution_steps):
            for bits_per_layer in range(0, number_of_bits_per_layer):
                X[i, steps, bits_per_layer] = int(bit[bit_count])
                bit_count += 1
    Y = datay[:600]
    X, Y = shuffle(X, Y)
    test_samples = 10
    x, x_test, y, y_test = X[0:-test_samples, :, :], X[-test_samples:-1, :, :], Y[0:-test_samples, :], Y[
                                                                                                       -test_samples:-1,
                                                                                                       :]
    return x,y, x_test, y_test

def call_main(datax, datay):
    x, y, x_test, y_test = load_task_data(datax, datay)
    bayesian_optimisation(x,y,x_test,y_test)

def load_task_data(datax, datay):
    X = np.loadtxt(datax, delimiter=" ")
    Y = np.loadtxt(datay, delimiter=" ")
    test_index = int(0.1*X.shape[0])
    x, x_test, y, y_test = X[:-test_index], X[-test_index:], Y[:-test_index], Y[-test_index:]
    return x,y,x_test,y_test