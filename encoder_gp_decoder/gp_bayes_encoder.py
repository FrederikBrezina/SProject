"""
Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from encoder_gp_decoder.dense_loss import loss_nn_dense
from encoder_gp_decoder.normal_RNN_gp import train_model, train_all_models, transform_into_timeseries
import sys
import pickle

reverse_order = True

def expected_improvement(x, gaussian_process, loss_optimum, greater_is_better=0, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """


    x_to_predict = x.reshape(-1, n_params)



    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)



    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] = 0.0


    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=0,
                               bounds=None, n_restarts=30, not_follow_bounds = True):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]
    count = 0
    loss_optimum = 0
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    methods = ['L-BFGS-B', 'BFGS']
    if not_follow_bounds == True:
        method = methods[1]
        bounds_temp = None
    else:
        method = methods[0]
        bounds_temp = bounds



    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):




        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds_temp,
                       method=method,
                       args=(gaussian_process, loss_optimum, greater_is_better, n_params))




        if count==0:
            best_acquisition_value = res.fun
            best_x = res.x

        elif res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

        count+=1
    return best_x


def bayesian_optimisation(x,y,x_test,y_test, act_fce, loss, optimizer, batch_size, min_depth, max_depth, min_units, max_units, n_iters,  n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=0.04, epsilon=1e-8, retrain_model_rounds = 50, greater_is_better=1):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    #This is the variable dense NN configuration architecture
    sample_loss = loss_nn_dense

    #Intial data
    x_list = []
    serialized_arch_list = []
    y_list = []
    decoded_sanitized_list, performance_metrics_list,  = [], []
    n_of_act_fce = len(act_fce)
    dimension_of_hidden_layers = 6  #this is the dimension between encoder decoder, also the dimension in which GP is working on
    bounds = np.zeros((dimension_of_hidden_layers, 2))
    bounds[:, 0] = -1
    bounds[:, 1] = 1
    n_params = bounds.shape[0]
    non_sense = False
    yp_list = []

    ##Train encoder decoder
    encoder, decoder, full_model, [datax_hidden_perf, datax_hidden_t_perf,
                                   datax_fce_perf, datax_fce_t_perf,decoded_sanitized_list,
                                   serialized_arch_list, performance_metrics_list] = train_model(
        x, y, x_test, y_test, act_fce,loss, optimizer,
        dimension_of_hidden_layers, n_of_act_fce, min_units, max_units,
        min_depth, max_depth, 1000, n_of_act_fce + 1, y.shape[1],
        reverse_order=reverse_order, initial_search=n_pre_samples)

    assert len(performance_metrics_list) == len(decoded_sanitized_list)

    encoded_data = encoder.predict([datax_hidden_perf, datax_fce_perf])
    assert len(encoded_data) == len(performance_metrics_list)

    x_list.extend(encoded_data)




    for i in range(0, len(performance_metrics_list)):
        yp_list.append(performance_metrics_list[i][greater_is_better])
        y_list.append(performance_metrics_list[i][greater_is_better])


    xp = np.array(x_list)
    yp = np.array(y_list)



    # Create the GP
    ##For possible gp_params configuration
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    #Else use default params and Matern kernel
    else:
        kernel = gp.kernels.Matern(length_scale=0.25)
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=50,
                                            normalize_y=True)

    #Now choose next architecture based on knowledge of past results
    n=0
    retrain_flag = True
    while n <n_iters:
        if (n%retrain_model_rounds == retrain_model_rounds-1) and retrain_flag:
            print(len(decoded_sanitized_list), len(performance_metrics_list))
            retrain_flag=False

            x_list = retrain_encode_again(decoded_sanitized_list, performance_metrics_list, encoder)
            xp = np.array(x_list)
        print("NN_after_intial search: ", n)
        print(len(x_list), len(performance_metrics_list), len(y_list), yp.shape[0])

        #Fit, the results into gp
        model.fit(xp, yp)



        # Sample next hyperparameter
        next_sample = sample_next_hyperparameter(expected_improvement, model, yp,
                                                     greater_is_better=greater_is_better, bounds=bounds, n_restarts=100, not_follow_bounds=False)


        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        the_same = True
        while the_same:
            # Decode and sanitize the example
            next_sample1 = next_sample.reshape((1, n_params))
            decoded_sanitized = sanitize_next_sample_for_gp(decoder.predict(next_sample1), n_of_act_fce + 1, min_units,
                                                            max_units, y.shape[1])
            for i2 in range(0, len(decoded_sanitized_list)):
                if np.array_equal(decoded_sanitized_list[i2], decoded_sanitized):
                    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                    the_same = True
                    break
                else:
                    the_same = False



        #Decode and sanitize the example
        datax_hidden_perf, datax_hidden_t_perf, datax_fce_perf, datax_fce_t_perf = transform_into_timeseries(
            [decoded_sanitized, ])

        encoded_data = encoder.predict([datax_hidden_perf, datax_fce_perf])

        x_list.extend(encoded_data)

        #If it satisfies the depth requirements, proceed to train it
        if decoded_sanitized[0] > 0:
            serialized_arch_list.append(seriliaze_next_sample_for_loss_fce(decoded_sanitized, n_of_act_fce + 1))
            retrain_flag = True
            # Sample loss for new set of parameters
            cv_score = sample_loss(serialized_arch_list[-1], x, y, x_test, y_test, act_fce, loss, optimizer, batch_size)

            decoded_sanitized_list.append(decoded_sanitized)
            performance_metrics_list.append(cv_score)
            print(serialized_arch_list[-1], cv_score[1])
            cv_score = cv_score[greater_is_better]
            yp_list.append(cv_score)
            y_list.append(cv_score)
            np.savetxt("performance1.txt", y_list)
            output = open('arch_list.pkl', 'wb')
            pickle.dump(serialized_arch_list, output)
            output.close()
            n += 1

        #If it does not, do not train, but set the cv_score to very high
        else:
            if greater_is_better:
                if non_sense:
                    cv_score = min(y_list)
                    y_list.append(cv_score)
                else:
                    cv_score = min(y_list)
                    non_sense = True
                    y_list.append(cv_score)
            else:
                if non_sense:
                    cv_score = max(y_list)
                    y_list.append(cv_score)
                else:
                    cv_score = max(y_list)
                    non_sense = True
                    y_list.append(cv_score)


        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    #Return the array in form (no_units, index_of_act, no_units....) and return their losses
    return serialized_arch_list, yp_list



def seriliaze_next_sample_for_loss_fce(next_sample, number_of_parameters_per_layer):
    ###Translate the array into array of alternating numbers.
    ###The alternating numbers are number of hidden units in layer, the index of activation in act_fce array
    ###Act fce array is the default array or the one given to through the command line interface

    seriliezed_next_sample = []
    next_sample = next_sample.tolist()
    number_of_layers = int((len(next_sample))/number_of_parameters_per_layer)

    for i in range(0, number_of_layers):
        #Append the number of hidden units
        seriliezed_next_sample.append(round(next_sample[i*number_of_parameters_per_layer]))
        #Append the index of activation
        #Search through the rest of the layer represented in the array
        #The rest contains a number which corresponds to unnormalized probability of choosing the activation
        seriliezed_next_sample.append(next_sample[(i*number_of_parameters_per_layer) + 1: (i+1)*number_of_parameters_per_layer]
                                      .index(max(next_sample[(i*number_of_parameters_per_layer) + 1: (i+1)*number_of_parameters_per_layer])))

    return np.array(seriliezed_next_sample)


def sanitize_next_sample_for_gp(next_sample, number_of_parameters_per_layer, min_units, max_units, dimension_of_out_put):
    ##This function serves as a first sanitization of the decoder output
    #Decoder outputs real numbers therefore we have to round them
    #Decoder as well outputs unnormalized porbability across the activations to use for each layer
    #This has to be sanitized as well, therefore the activation function with highest number is set to 1, rest to 0
    depth = next_sample[0].shape[1]
    seriliezed_next_sample = np.zeros((number_of_parameters_per_layer*depth))
    #The decoded output is timedistributed in 3rd dimension, flatten it


    if reverse_order:
        for i in range(0, depth):

            #This is the number of hidden units

            temp = round(next_sample[0][0,i ,0])

            if (temp < 0.5) and (i==0):
                return np.zeros((number_of_parameters_per_layer))
            #If it predicts less than 0.5 units than this means the NN config reached its depth
            elif temp < 0.5:
                #Return the shortened example
                seriliezed_next_sample = seriliezed_next_sample[:i * number_of_parameters_per_layer]
                return seriliezed_next_sample

            #If the depth is maximum hardcode the dimension of output
            if i == 0:
                temp = dimension_of_out_put
            seriliezed_next_sample[i * number_of_parameters_per_layer] = temp

            next_sample_temp = next_sample[1][0,i].tolist()
            #Find the index of maxium of unnoramlized porbabilities of activations
            index = next_sample_temp.index(max(next_sample_temp))

            for fce in range(0, number_of_parameters_per_layer - 1):
                #Set the maximum to one
                if index == fce:
                    seriliezed_next_sample[i * number_of_parameters_per_layer + fce + 1] = 1
                #Set rest to 0
                else:
                    seriliezed_next_sample[i * number_of_parameters_per_layer + fce + 1] = 0

    return seriliezed_next_sample

def retrain_encode_again(decoded_sanitized_list, performance_metrics_list, encoder):
    train_all_models(decoded_sanitized_list, performance_metrics_list)

    encoded_data_list = []
    datax_hidden_perf, datax_hidden_t_perf, datax_fce_perf, datax_fce_t_perf = transform_into_timeseries(
        decoded_sanitized_list[:])

    encoded_data = encoder.predict([datax_hidden_perf, datax_fce_perf])
    assert len(encoded_data) == len(decoded_sanitized_list)
    encoded_data_list.extend(encoded_data)
    return encoded_data_list



