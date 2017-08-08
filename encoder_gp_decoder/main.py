import pickle
import numpy as np
from encoder_gp_decoder.gp_bayes_encoder import bayesian_optimisation
import sys, pickle

def load_task_data(datax, datay, test_number):
    X = np.loadtxt(datax, delimiter=" ")[:500]
    Y = np.loadtxt(datay, delimiter=" ")[:500]
    test_index = test_number
    x, x_test, y, y_test = X[:-test_index], X[-test_index:], Y[:-test_index], Y[-test_index:]
    return x,y,x_test,y_test

def call_main(loss, optimizer, min_depth, max_depth, min_units, max_units, act_fce, n_iter,
              n_presamples, datasetx, datasety, test_number, output,cv_score, batch_size):
    #Load the tasak data
    x, y, x_test, y_test = load_task_data(datasetx, datasety, test_number)
    #Find the best architectures
    arch_list, loss_list = bayesian_optimisation(x,y,x_test,y_test, act_fce, loss, optimizer, batch_size, min_depth,
                                                 max_depth, min_units, max_units, n_iter, n_pre_samples=n_presamples)
    #Save the NN_configurations
    output1 = open(output, 'wb')
    pickle.dump(arch_list, output1)
    output1.close()
    #Save their lowest test_losses
    output2 = open(cv_score, 'wb')
    pickle.dump(arch_list, output2)
    output2.close()

    print(arch_list,loss_list)

def command_line_interface():
    ##getting activation functions to search trough
    act_fce = []
    act_fce_keras = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                     'linear']
    print('Do yoy want to include all these layers in search? n/y')
    print(act_fce_keras)
    j = input()
    if j == 'n':
        print('Write down keras activations one by one, if you have no more functions tot include wirte n')
        goon = True
        while goon:
            f = input()
            if f=='n':
                goon = False
                print('Write optimizer as keras name')
            else:
                act_fce.append(f)
    elif j == 'y':
        act_fce = act_fce_keras
        print('Write optimizer as keras name')
    else:
        print('Have not recognized input')
        sys.exit()
    ##########################################
    #Optimizer
    optimizer = input()
    ##########################################
    #loss
    print('Loss')
    loss = input()
    ##########################################
    #min depth
    print('minimum number of layers? enter number')
    min_depth = int(input())
    ##########################################
    #max_depth
    print('maximum number of layers? enter number')
    max_depth = int(input())
    #########################################
    print('minimum number of hidden units per layer? enter number')
    min_units = int(input())
    #########################################
    print('maximum number of hidden units per layer? enter number')
    max_units = int(input())
    ##########################################
    #number_of_iterations
    print('number of trained nns after initial search, enter number')
    n_iter = int(input())
    ##########################################
    #number of nns trained per layer as initial search
    print('number of nns trained as initial search, enter number')
    n_presamples = int(input())
    ##########################################
    #datasetx
    print('enter path to input dataset X')
    datasetx = input()
    ##########################################
    # datasety
    print('enter path to output dataset Y')
    datasety = input()
    ##########################################
    #number of test sets
    print('How many examples to use for test set from the data set, enter number')
    test_number = int(input())
    ##########################################
    #where to save output
    print('enter path where to save output architectures')
    output = input()
    ##########################################
    #where to save loss list
    print('where to save losses acquired for the strcutres')
    cv_score = input()
    ############################################
    print('batch_size?')
    batch_size = int(input())


    return act_fce, optimizer, loss, min_depth, max_depth, min_units, max_units, n_iter,\
           n_presamples, datasetx, datasety, test_number, output, cv_score, batch_size


if __name__ == "__main__":
    # act_fce, optimizer, loss, min_depth, max_depth, min_units, max_units, n_iter, n_presamples, datasetx, datasety, test_number, output,cv_score, batch_size = command_line_interface()
    loss, optimizer, min_depth, max_depth, min_units, max_units, act_fce, n_iter, n_presamples, \
    datasetx, datasety, test_number, output, cv_score, batch_size = 'categorical_crossentropy', 'adam', 2, 11, 2, 100, [
        'softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
        'linear'], 100, 20, 'X_basic_task.txt', 'Y_basic_task.txt', 100, 'arch.txt', "loss.txt", 10
    call_main(loss, optimizer, min_depth, max_depth, min_units, max_units, act_fce, n_iter,
              n_presamples, datasetx, datasety, test_number, output, cv_score,batch_size)
