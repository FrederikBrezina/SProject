import numpy as np

import HelpFunctions as hp
import tensorflow as tf

def calculate(layers_second_derivative, weights_array, convergence_time_step):
    thread_list = []
    for layers in range(0, len(weights_array)):
        layers_second_derivative[layers,:] = second_order(weights_array,convergence_time_step,layers)



def second_order (weights_array,convergence_time_step,layer_index):

    a = tf.constant(weights_array[layer_index][0:-2,:,:])
    b = tf.constant(weights_array[layer_index][1:-1,:,:])
    s = tf.subtract(b,a)
    s = tf.subtract(s[1:-1,:,:],s[0:-2,:,:])
    results = [0, 0 ,0 ,0]

    square = tf.square(s)
    results[0] = tf.reduce_sum(square[0:convergence_time_step-1,:,:])
    results[1] = tf.reduce_sum(square[convergence_time_step-1:-1,:,:])

    cubic = tf.multiply(s,s)
    cubic = tf.multiply(s, cubic)
    results[2] = tf.reduce_sum(cubic[0:convergence_time_step-1,:,:])
    results[3] = tf.reduce_sum(cubic[convergence_time_step-1:-1,:,:])
    ses = tf.Session()
    results = ses.run(results)

    return results
