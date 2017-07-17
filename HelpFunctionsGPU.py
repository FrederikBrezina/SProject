import numpy as np
from multiprocessing import Pool
import HelpFunctions as hp

def calculate(layers_second_derivative, weights_array, convergence_time_step):
    thread_list = []
    for layers in range(0, len(weights_array)):
        for axis_0 in range(0, weights_array[layers].shape[1]):
            for axis_1 in range(0, weights_array[layers].shape[2]):
                thread_list.append([weights_array, convergence_time_step, layers, axis_0, axis_1])
    with Pool() as p:
        results = p.starmap(run, thread_list)
    for result in results:
        layers_second_derivative[result[0]][:] += result[1:len(result)]

def run(weights_array,convergence_time_step,layer_index, unit_index_axis_0, unit_index_axis_1 ):
    temp_before_conv = hp.smooth_the_data_moving_average(weights_array[layer_index][0:convergence_time_step, unit_index_axis_0, unit_index_axis_1],
                                                         240)
    temp2_before_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_before_conv), 50)
    a = np.sum(np.absolute(temp2_before_conv))
    b = np.sum(temp2_before_conv)
    temp_after_conv = hp.smooth_the_data_moving_average(weights_array[layer_index][convergence_time_step:-1, unit_index_axis_0, unit_index_axis_1],
                                                        240)
    temp2_after_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_after_conv), 50)
    c = np.sum(np.absolute(temp2_after_conv))
    d = np.sum(temp2_after_conv)

    #calculated squared sum
    e = np.sum(np.square(temp2_before_conv))
    return [layer_index, a , b ,c ,d , e]