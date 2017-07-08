import numpy as np
from threading import Thread
import HelpFunctions as hp
class LayersThreading():
    def __init__(self, layers_second_derivative, weights_array, convergence_time_step):
        self.layers_second_derivative ,self.weights_array, self.convergence_time_step = layers_second_derivative, weights_array, convergence_time_step
    def calculate(self):
        thread_list = []
        for layers in range(0, len(self.weights_array)):
            for axis_0 in range(0, self.weights_array[layers].shape[1]):
                for axis_1 in range(0, self.weights_array[layers].shape[2]):
                    thread_list.append(UnitsThreading(self.layers_second_derivative, self.weights_array, self.convergence_time_step, layers, axis_0, axis_1))
                    thread_list[-1].start()
        for thread in thread_list:
            thread.join()

class UnitsThreading(Thread):
    def __init__(self,layers_second_derivative,weights_array, convergence_time_step, layer_index, unit_index_axis_0, unit_index_axis_1):
        super(UnitsThreading, self).__init__()
        self.layers_second_derivative, self.weights_array, self.convergence_time_step, self.layer_index, self.unit_index_axis_0, self.unit_index_axis_1 = layers_second_derivative, weights_array, convergence_time_step, layer_index, unit_index_axis_0, unit_index_axis_1
    def run(self):
        temp_before_conv = hp.smooth_the_data_moving_average(self.weights_array[self.layer_index][0:self.convergence_time_step, self.unit_index_axis_0, self.unit_index_axis_1],
                                                             240)
        temp2_before_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_before_conv), 50)
        self.layers_second_derivative[self.layer_index, 0] += np.sum(np.absolute(temp2_before_conv))
        self.layers_second_derivative[self.layer_index, 1] += np.sum(temp2_before_conv)
        temp_after_conv = hp.smooth_the_data_moving_average(self.weights_array[self.layer_index][self.convergence_time_step:-1, self.unit_index_axis_0, self.unit_index_axis_1],
                                                            240)
        temp2_after_conv = hp.smooth_the_data_moving_average(hp.second_order_derivate(temp_after_conv), 50)
        self.layers_second_derivative[self.layer_index, 2] = np.sum(np.absolute(temp2_after_conv))
        self.layers_second_derivative[self.layer_index, 3] = np.sum(temp2_after_conv)

        #calculated squared sum
        self.layers_second_derivative[self.layer_index, 4] += np.sum(np.square(temp2_before_conv))
