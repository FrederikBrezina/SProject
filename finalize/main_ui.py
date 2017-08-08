import finalize.gp_bayes_var_depth_finished as bayes_var
from finalize.dense_training import loss_nn_dense
import numpy as np
import sys

def call_core(loss_fce,optimizer,min_depth, max_depth,min_units, max_units, act_functions_to_use, number_of_iteration, number_of_pre_samples_per_depth, datasetx, datasety, number_of_test_examples, output):
    ###Load dataset
    datax, datay = np.loadtxt(datasetx, delimiter=" "), np.loadtxt(datasety, delimiter=" ")
    x, x_test = datax[:(-1-number_of_test_examples)], datax[(-1-number_of_test_examples):-1]
    x = x[:300]
    y, y_test = datay[:(-1 - number_of_test_examples)], datay[(-1 - number_of_test_examples):-1]
    y = y[:300]
    number_of_act_fce = len(act_functions_to_use)
    bounds = np.zeros((int((max_depth-1)*(number_of_act_fce+1) + number_of_act_fce),2))

    #Last layer is special as output shape is given
    for i in range(0,max_depth - 1):
        bounds[i*(number_of_act_fce + 1),0] = min_units
        bounds[i*(number_of_act_fce + 1),1] = max_units
        for act_bounds in range(0, number_of_act_fce):
            bounds[i * (number_of_act_fce + 1) + 1 + act_bounds, 0] = 0
            bounds[i * (number_of_act_fce + 1) + 1 + act_bounds, 1] = 1

    for act_bounds in range(int((max_depth-1)*(number_of_act_fce+1)),int((max_depth-1)*(number_of_act_fce+1) + number_of_act_fce)):
        bounds[act_bounds, 0] = 0
        bounds[act_bounds, 1] = 1

    return bayes_var.bayesian_optimisation(x,y, x_test, y_test, loss_fce, optimizer,min_depth, max_depth,
                                           number_of_iteration,loss_nn_dense, bounds,act_functions_to_use,
                                           n_pre_samples=number_of_pre_samples_per_depth, output=output)

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
    print('number of nns trained per layer as initial search, enter number')
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
    print('enter path where to save output')
    output = input()


    return act_fce, optimizer, loss, min_depth, max_depth, min_units, max_units,\
           n_iter, n_presamples, datasetx, datasety, test_number, output

if __name__ == "__main__":
    # act_fce, optimizer, loss, min_depth, max_depth, min_units, max_units, n_iter, n_presamples, datasetx, datasety, test_number, output = command_line_interface()
    loss, optimizer, min_depth, max_depth, min_units, max_units, act_fce, n_iter, n_presamples,\
    datasetx, datasety, test_number, output = 'categorical_crossentropy', 'adam', 2,10,2,100, \
                                              ['softmax', 'elu',  'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid','linear'] ,\
                                              10,2,'X.txt', 'Y.txt', 10,'c.txt'
    call_core(loss, optimizer, min_depth, max_depth, min_units, max_units, act_fce, n_iter,
              n_presamples, datasetx, datasety, test_number, output)




