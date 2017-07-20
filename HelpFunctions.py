import numpy as np
def convergence_of_NN_val_loss(val_loss, convergence_rate = 4):
    #Smaller the better
    #Find the best model

    index = val_loss.index(min(val_loss))
    #Take into account only the curve before the best model
    val_loss1 = val_loss[0:(index +1)]
    #Find the lowest point before the best model
    #The number before the best model to search through
    #is convergence rate
    x = val_loss[-1]
    leng = len(val_loss1)
    if leng<5 :
        return 1
    for i in range (2, (convergence_rate+2)):
        if val_loss1[-i]>val_loss1[1-i]:
            x = val_loss1[-i]
    diff = (x - val_loss1[-1])/(2.75)
    #clip the diff in case
    # if diff>0.07:
    #     diff = 0.07
    for i in range(0, len(val_loss)):
        if (val_loss[i] - diff) <= val_loss1[-1]:
            return (i+1)

    return leng

def highest_accuracy_of_NN(val_acc):
    return max(val_acc)
def lowest_val_loss(val_loss):
    return min(val_loss)
def mean_and_std(values_list):
    length = len(values_list)
    average = 0
    std = 0
    for i in range(0, length):
        average += values_list[i]
    average /= length
    for i in range(0, length):
        std += (values_list[i] - average)**2
    std /= length - 1
    std = std**0.5
    return [average, std]

def mean_and_std_of_list_of_lists(values_list):
    length = len(values_list)
    avg_list = []
    std_list = []
    avg_peaks_list = []
    std_peaks_list = []
    for i in range(0,length):
        avg1, std1 = std_in_time_series(values_list[i])
        avg2, std2 = std_of_peaks_in_time_series(values_list[i])
        avg_list.append(avg1) , std_list.append(std1), avg_peaks_list.append(avg2), std_peaks_list.append(std2)

    avg_mean, avg_std = mean_and_std(avg_list)
    std_mean, std_std = mean_and_std(std_list)
    avg_peaks_mean, avg_peaks_std = mean_and_std(avg_peaks_list)
    std_peaks_mean, std_peaks_std = mean_and_std(std_peaks_list)
    return [avg_mean, avg_std, std_mean, std_std, avg_peaks_mean, avg_peaks_std, std_peaks_mean, std_peaks_std]

def mean_and_std_of_performance_list_of_lists(values_list):
    perf_list = []
    for i in range(0,len(values_list)):
        perf_list.append(max(values_list[i]))
    return mean_and_std(perf_list)

def rate_of_list(values_list):
    rate_list = []
    for i in range(0, len(values_list) - 1):
        rate_list.append((values_list[i+1] - values_list[i]))
    return rate_list

def skewness(values_list):
    skewness = 0
    length = len(values_list)
    average, std = mean_and_std(values_list)
    for i in range(0, length):
        skewness += (values_list[i] - average) ** 3
    skewness /= length - 1
    skewness /= std**3
    return skewness

def kurtosis(values_list):
    kurtosis = 0
    length = len(values_list)
    average, std = mean_and_std(values_list)
    for i in range(0, length):
        kurtosis += (values_list[i] - average) ** 4
    kurtosis /= length - 1
    kurtosis /= std**4
    return kurtosis

def std_in_time_series(value_list):
    #Epochs are x axis order in the list
    #For each epoch there is probability distribution as list
    #The x-axis is taken from range 1 - len(value_list)
    #Use for normalized list like accuracy (as every model will have maximum accuracy of 100)
    length = len(value_list)
    total_sum = 0
    std = 0
    avg = 0
    for i in range(0,length):
        total_sum += value_list[i]
        avg += value_list[i]*(i+1)

    avg /= total_sum

    for i in range(0,length):
        std += value_list[i]*((i + 1 - avg)**2)

    std /= total_sum
    std = std**0.5
    return [avg , std]



def overfitting_all_values(list_of_loss, list_of_val_loss, conv_epoch):
    diff_list = []
    diff_list.append(0)
    diff_list.append(0)
    for epoch in range(0, len(list_of_loss)):
        diff = list_of_val_loss[epoch] - list_of_loss[epoch]
        diff_list.append(diff)
    #Adujusting conv_epoch by two as second order derivative eats 2 more points
    if conv_epoch > 2:
        conv_epoch_second = conv_epoch - 2
    else:
        conv_epoch_second = conv_epoch

    #Calculate second order things
    second_order_dev = second_order_derivate(diff_list)
    second_order_before = second_order_dev[0:conv_epoch_second]
    second_order_after = second_order_dev[conv_epoch_second:len(second_order_dev)]
    sum_of_second_order_before, sum_of_absolute_second_order_before, sum_of_second_order_after = 0 , 0, 0
    for point in range(0, len(second_order_before)):
        sum_of_absolute_second_order_before += abs(second_order_before[point])
        sum_of_second_order_before += second_order_before[point]
    sum_of_second_order_after = sum(second_order_after)

    #Calculate maximum diff, and diff at conv point
    diff_at_conv = diff_list[conv_epoch - 1]
    max_diff = max(diff_list)

    return [sum_of_second_order_before, sum_of_absolute_second_order_before, sum_of_second_order_after, diff_at_conv, max_diff]



def second_order_derivate(list_of_values):
    return_list = first_order_derivative(list_of_values)
    return_list = first_order_derivative(return_list)
    return return_list

def first_order_derivative(list_of_values):
    return_list = []
    for grad in range(1, len(list_of_values)):
        return_list.append(list_of_values[grad] - list_of_values[grad - 1])
    return return_list

def smooth_the_data_moving_average(list_of_values, range_of_average = 70):
    smoothed_data = []
    half_range_of_average=int(np.rint(range_of_average/2))
    num_of_points_in_average = 2 * half_range_of_average + 1
    for point in range(half_range_of_average, len(list_of_values) - half_range_of_average - 1):
        sum = 0
        for running_avg in range(point - half_range_of_average, point + half_range_of_average + 1):
            sum += list_of_values[running_avg]
        sum /= num_of_points_in_average
        smoothed_data.append(sum)
    return smoothed_data

#return data with same average and standard deviation
def return_smoothed_data_with_average_std_given(tensor_to_calc, avg = 0, std = 1):
    shape = tensor_to_calc.shape
    return_tensor = np.copy(tensor_to_calc)
    total_num_of_points = 1
    average = 0
    std_act = 0
    #Calculate the total number of units
    for dimension in range(0, len(shape)):
        total_num_of_points *= shape[dimension]
    #calculate average
    for element in np.nditer(tensor_to_calc):
        average += element
    average /= total_num_of_points
    #Calculate standard deviation
    for element in np.nditer(tensor_to_calc):
        std_act += (element - average)**2
    std_act /= total_num_of_points - 1
    std_act = std_act**0.5
    #Adjust each element
    for element in np.nditer(return_tensor , op_flags=['readwrite']):
        element[...] = element - average
        element[...] = element[...]*std/std_act
        element[...] = element[...] + avg
    return return_tensor


#########################################################################
###########################################################################
###########################################################################
#Definitions which are now redundant
def handle_weight_variance_for_layers(list_of_weight_matrices):
    weight_data = []
    #set up the empty list
    for layers in range(0,len(list_of_weight_matrices)):
        weight_data.append(np.zeros_like(list_of_weight_matrices[layers][0]))
    #Calculate the data
    for layers in range(0,len(list_of_weight_matrices)):
        for axis_0 in range(0, list_of_weight_matrices[layers][0].shape[0]):
            for axis_1 in range(0, list_of_weight_matrices[layers][0].shape[1]):
                last_diff = list_of_weight_matrices[layers][0][axis_0][axis_1] - list_of_weight_matrices[layers][1][axis_0][axis_1]
                for time_step in range(1, len(list_of_weight_matrices[layers]) - 1):
                    act_diff = list_of_weight_matrices[layers][time_step][axis_0][axis_1] - list_of_weight_matrices[layers][time_step + 1][axis_0][axis_1]
                    if np.sign(last_diff) != np.sign(act_diff):
                        weight_data[layers][axis_0][axis_1] += 1
                    last_diff = act_diff


    return weight_data

def avg_and_std_of_overfitting(traing_acc_list_of_lists, validation_acc_list_of_lists , convergence_points_list):
    diff_list_of_lists = []

    #Calculate the difference between training accuracy and validation accuracy
    for i in range(0,len(traing_acc_list_of_lists)):
        diff_list = []
        for i2 in range(0,len(traing_acc_list_of_lists[i])):
            diff_list.append(traing_acc_list_of_lists[i][i2] - validation_acc_list_of_lists[i][i2])
        diff_list_of_lists.append(diff_list)

    #Calculate the values
    #If the difference does not fall below this value overfitting started from last point where it
    #did fall below
    index_list = []
    rate_list = []
    for i in range(0, len(traing_acc_list_of_lists)):
        #Set default values if overfitting did not happen
        index = len(traing_acc_list_of_lists[i])
        rate = 0
        line = 0
        #find the line for i
        for i2 in range(0, convergence_points_list[i]):
            line += diff_list_of_lists[i][i2]
        line /= convergence_points_list[i]
        #find the last point where it does fall down
        if line != 0:
            index = convergence_points_list[i] -1
            for i2 in range(convergence_points_list[i] - 1, len(traing_acc_list_of_lists[i])):
                if diff_list_of_lists[i][i2] < line :
                    index = i2
            index += 1
            points_in_overfit = len(traing_acc_list_of_lists[i]) - index + 1
            rate = diff_list_of_lists[i][-1] - diff_list_of_lists[i][index-1]
            rate /= points_in_overfit

        index_list.append(index)
        rate_list.append(rate)
    return [mean_and_std(index_list) , mean_and_std(rate_list)]

def std_of_peaks_in_time_series(value_list):
    #Epochs are x axis order in the list
    #For each epoch there is probability distribution as list
    #The x-axis is taken from range 1 - len(value_list)
    #Use for normalized list like accuracy (as every model will have maximum accuracy of 100)
    length = len(value_list)
    total_sum = 0
    std = 0
    avg = 0
    for i in range(0,length):
        sign = value_list[i] / abs(value_list[i])
        total_sum += sign*value_list[i]**2
        avg += sign*(value_list[i]**2)*(i+1)

    avg /= total_sum

    for i in range(0,length):
        sign = value_list[i] / abs(value_list[i])
        std += sign*(value_list[i]**2)*((i + 1 - avg)**2)

    std /= total_sum
    std = std**0.5
    return [avg , std]