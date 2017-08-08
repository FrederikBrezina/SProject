""" gp.py

Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

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
    boundsshape = bounds.shape[0]

    count = 0
    for starting_point in np.random.uniform(bounds[:boundsshape, 0], bounds[:boundsshape, 1], size=(n_restarts,boundsshape)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, boundsshape))
        if count==0:
            best_acquisition_value = res.fun
            best_x = res.x

        elif res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x
        count+=1
    return best_x, best_acquisition_value


def bayesian_optimisation(x, y, x_test, y_test, loss_fce, optimizer, min_depth, max_depth, n_iters, sample_loss, bounds, act_fce, output, x0=None, n_pre_samples=5,
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
    x_list = []
    xp_list = []
    y_list = []
    models = []
    xp, yp = None, None
    n_params = bounds.shape[0]
    n_params_per_layer = len(act_fce) + 1
    max_depth = int(max_depth)
    for depth in range(0, (max_depth - min_depth + 1)):
        x_list.append([])
        y_list.append([])


    for depth in range(0, max_depth - min_depth + 1):
        act_depth = depth*n_params_per_layer + (min_depth - 1)*n_params_per_layer
        for params in np.random.uniform(bounds[:act_depth, 0], bounds[:act_depth, 1], size=(n_pre_samples, act_depth)):
            # Handle data where discrete needs to be
            params2 = np.zeros((act_depth+len(act_fce)))
            params2[:act_depth] = params
            params2[act_depth:] = np.random.uniform(bounds[-len(act_fce):, 0], bounds[-len(act_fce):, 1], size=len(act_fce))
            params2 = serialize_next_sample_for_gp(params2, n_params_per_layer)
            x_list[depth].append(params2)
            # print(x_list[depth])
            y_list[depth].append(sample_loss(seriliaze_next_sample_for_loss_fce(params2, n_params_per_layer), act_fce, x, y, x_test, y_test, loss_fce, optimizer))
        xp = np.array(x_list[depth])
        yp = np.array(y_list[depth])
        xp_list.append(xp)

        # Create the GP
        if gp_params is not None:
            models.append(gp.GaussianProcessRegressor(**gp_params))
        else:
            kernel = gp.kernels.Matern()
            models.append(gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=10,
                                                    normalize_y=True))
        models[-1].fit(xp, yp)

    next_sample_list, loss_list = [], []
    depth_to_search_through = None
    for n in range(n_iters):

        if n==0:
            for depth in range(0,max_depth - min_depth + 1):
                act_depth = (depth + min_depth - 1)*n_params_per_layer
                bounds_depth = np.zeros((act_depth + len(act_fce), 2))
                bounds_depth[:act_depth] = bounds[:act_depth]
                bounds_depth[act_depth:] = bounds[-len(act_fce):]
                next_sample, loss = sample_next_hyperparameter(expected_improvement, models[depth], np.array(y_list[depth]), greater_is_better=False, bounds=bounds_depth, n_restarts=100)
                loss_list.append(loss)

                # Handle data where discrete needs to be

                next_sample = serialize_next_sample_for_gp(next_sample, n_params_per_layer)

                # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                if np.any(np.abs(next_sample - xp_list[depth]) <= epsilon):
                    next_sample = np.random.uniform(bounds_depth[:,0], bounds_depth[:, 1], act_depth + len(act_fce))
                    next_sample = serialize_next_sample_for_gp(next_sample, n_params_per_layer)
                next_sample_list.append(next_sample)
        else:
            act_depth = (depth_to_search_through + min_depth - 1) * n_params_per_layer
            bounds_depth = np.zeros((act_depth + len(act_fce), 2))
            bounds_depth[:act_depth] = bounds[:act_depth]
            bounds_depth[act_depth:] = bounds[-len(act_fce):]
            next_sample, loss = sample_next_hyperparameter(expected_improvement, models[depth_to_search_through], np.array(y_list[depth_to_search_through]),
                                                           greater_is_better=False, bounds=bounds_depth, n_restarts=100)
            loss_list[depth_to_search_through] = loss

            # Handle data where discrete needs to be

            next_sample = serialize_next_sample_for_gp(next_sample, n_params_per_layer)

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            if np.any(np.abs(next_sample - xp_list[depth_to_search_through]) <= epsilon):
                next_sample = np.random.uniform(bounds_depth[:, 0], bounds_depth[:, 1], act_depth + len(act_fce))
                next_sample = serialize_next_sample_for_gp(next_sample, n_params_per_layer)
            next_sample_list[depth_to_search_through]=next_sample

        depth_to_search_through = loss_list.index(min(loss_list))
        # Sample loss for new set of parameters

        cv_score = sample_loss(seriliaze_next_sample_for_loss_fce(next_sample_list[depth_to_search_through], n_params_per_layer), act_fce, x, y, x_test, y_test, loss_fce, optimizer)

        # Update lists
        x_list[depth_to_search_through].append(next_sample_list[depth_to_search_through])
        y_list[depth_to_search_through].append(cv_score)

        # Update xp and yp
        xp = np.array(x_list[depth_to_search_through])
        yp = np.array(y_list[depth_to_search_through])
        xp_list[depth_to_search_through] = xp
        models[depth_to_search_through].fit(xp, yp)


    return xp, yp, y_list

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