""" gp.py

Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp
import copy
from scipy.stats import norm
from scipy.optimize import minimize
from rnn_gp_rnn.dense_loss import loss_nn_dense

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
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

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
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
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
        if count==0:
            best_acquisition_value = res.fun

        elif res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

        count+=1
    return best_x


def bayesian_optimisation(x,y,x_test,y_test, act_fce, loss, optimizer, batch_size, min_depth, max_depth, min_units, max_units, n_iters,   x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
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
    sample_loss = loss_nn_dense
    bounds = None
    x_list = []
    y_list = []
    n_of_act_fce = len(act_fce)
    n_params = bounds.shape[0]

    ##Determines the best last act_fce with first try
    indeces_act_fce = determine_the_best_last_act_fce(x, y, x_test, y_test, act_fce, 4, sample_loss, loss, optimizer, batch_size,
                                    other_layers=['tanh', 'relu'])
    #Tries again but just with top 3
    last_layer_top_3 = copy.deepcopy(act_fce)
    last_layer_top_3 = np.array(last_layer_top_3)
    last_layer_top_3 = last_layer_top_3[indeces_act_fce]
    last_layer_top_3 = last_layer_top_3[:3]
    indeces_act_fce_top_3 = determine_the_best_last_act_fce(x, y, x_test, y_test, last_layer_top_3, 4, sample_loss, loss, optimizer, batch_size,
                                    other_layers=['tanh', 'relu'])
    ##Index in act_fce of best last activation function of act_fce list
    index_of_best_last_layer = indeces_act_fce[indeces_act_fce_top_3[0]]


    for tri in range(0,n_pre_samples):
        depth = np.random.random()*(max_depth - min_depth + 1)*(1 - epsilon) + (min_depth - 0.5 + epsilon)
        ###Creating bounds without the last layer
        bounds = create_bounds(n_of_act_fce,min_units, max_units, depth, index_of_best_last_layer)


    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            params2 = serialize_next_sample_for_gp(params, n_of_act_fce + 1)
            x_list.append(params2)
            y_list.append(sample_loss(seriliaze_next_sample_for_loss_fce(params2, n_of_act_fce + 1), x, y, x_test, y_test, act_fce, loss, optimizer, batch_size, best_last_act = False))

    # else:
    #     for params in x0:
    #         x_list.append(params)
    #         y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1],  n_params)
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp

def create_bounds(num_of_act_fce, min_units, max_units, depth, index_of_best_last_layer_act):
    bounds = np.zeros(((depth - 1)*(1+ num_of_act_fce) + num_of_act_fce,2))
    for i in range(0, depth-1):
        bounds[i*(num_of_act_fce+1),0] = min_units
        bounds[i * (num_of_act_fce + 1), 1] = max_units
        for j in range(1, num_of_act_fce +1):
            bounds[i * (num_of_act_fce + 1) + j , 0] = 0
            bounds[i * (num_of_act_fce + 1) + j, 1] = 1
    #Last layer activation function, already determined, can be extended to multiple indeces
    for j in range(0, num_of_act_fce):
        if j == index_of_best_last_layer_act:
            bounds[(depth-1) * (num_of_act_fce + 1) + j, 1] = 1
            bounds[(depth - 1) * (num_of_act_fce + 1) + j, 0] = 0
        else:
            bounds[(depth - 1) * (num_of_act_fce + 1) + j, 1] = 0
            bounds[(depth - 1) * (num_of_act_fce + 1) + j, 0] = 0
    return bounds

def seriliaze_next_sample_for_loss_fce(next_sample, number_of_parameters_per_layer):
    seriliezed_next_sample = []
    next_sample = next_sample.tolist()
    number_of_layers = int((len(next_sample) - number_of_parameters_per_layer + 1)/number_of_parameters_per_layer)
    for i in range(0, number_of_layers):
        seriliezed_next_sample.append(round(next_sample[i*number_of_parameters_per_layer]))
        seriliezed_next_sample.append(next_sample[(i*number_of_parameters_per_layer) + 1: (i+1)*number_of_parameters_per_layer].index(max(next_sample[(i*number_of_parameters_per_layer) + 1: (i+1)*number_of_parameters_per_layer])))
    seriliezed_next_sample.append(next_sample[(number_of_layers*number_of_parameters_per_layer): (number_of_layers+1)*number_of_parameters_per_layer].index(max(next_sample[(number_of_layers*number_of_parameters_per_layer): (number_of_layers+1)*number_of_parameters_per_layer])))
    return np.array(seriliezed_next_sample)

def serialize_next_sample_for_gp(next_sample, number_of_parameters_per_layer):
    next_sample = next_sample.tolist()
    seriliezed_next_sample = np.copy(next_sample)
    number_of_layers = int((len(next_sample) - number_of_parameters_per_layer + 1) / number_of_parameters_per_layer)
    for i in range(0, number_of_layers):
        seriliezed_next_sample[i * number_of_parameters_per_layer] = round(next_sample[i * number_of_parameters_per_layer])
        index = next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer].index(
                max(next_sample[(i * number_of_parameters_per_layer) + 1: (i + 1) * number_of_parameters_per_layer]))
        for fce in range(1, number_of_parameters_per_layer):
            if index == fce:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce] = 1
            else:
                seriliezed_next_sample[i * number_of_parameters_per_layer + fce] = 0

    index = next_sample[(number_of_layers * number_of_parameters_per_layer): (number_of_layers + 1) * number_of_parameters_per_layer].index(
        max(next_sample[(number_of_layers * number_of_parameters_per_layer): (number_of_layers + 1) * number_of_parameters_per_layer]))
    for fce in range(0, number_of_parameters_per_layer - 1):
        if index == fce:
            seriliezed_next_sample[number_of_layers * number_of_parameters_per_layer + fce] = 1
        else:
            seriliezed_next_sample[number_of_layers * number_of_parameters_per_layer + fce] = 0
    return seriliezed_next_sample

def determine_the_best_last_act_fce(x,y,x_test,y_test,act_fce,depth,sample_loss,loss, optimizer, batch_size, other_layers = ['tanh', 'relu']):
    epsilon = 1e-7
    n_of_act_fce = len(act_fce)
    n_of_other_act_fce = len(other_layers)
    next_sample_act_fce = []
    for i in range(0, depth - 1):
        next_sample_act_fce.append(int(round(np.random.random()*(n_of_other_act_fce)*(1 - epsilon) - 0.5 + epsilon)))
    next_sample_temp = []
    remainder = 0
    gradient_of_units_decreasing = 20/depth
    losses = []
    for j in range(0, depth - 1):
        next_sample_temp.append(30 - int(remainder))
        next_sample_temp.append(next_sample_act_fce[j])
        remainder += gradient_of_units_decreasing
    for i in range(0, n_of_act_fce):
        next_sample = copy.deepcopy(next_sample_temp)
        next_sample.append(i)
        losses.append(sample_loss(next_sample,x,y,x_test,y_test,act_fce, loss, optimizer, batch_size,other_layers, best_last_act = True))

    indeces = sorted(range(len(losses)), key=lambda k: losses[k])
    return indeces


