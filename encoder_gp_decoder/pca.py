import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pickle
import sys
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


    if True:
        for i in range(0, depth):

            #This is the number of hidden units

            temp = round(next_sample[0][0,i ,0])

            if (temp < 0.5) and (i==0):
                print(next_sample[0][0,:,0])
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
x = """[[-0.25210184  0.99994922  0.85119146  0.57063794]
 [-0.27999598  0.97334027 -0.14743297 -0.94238412]
 [ 0.41043386  0.99114907  0.50098842  0.60373414]
 [ 0.40909371  0.9926362   0.52274692  0.61888951]
 [ 0.95542139 -0.99861449  0.99322748 -0.99920708]
 [ 0.97674942 -0.99992472  0.67264616 -0.99969381]
 [ 0.07299654  0.99930942  0.70327377  0.50696456]
 [ 0.9139021  -0.9949221   0.9902153  -0.99634689]
 [-0.22257094  0.98171878 -0.15625016 -0.95999295]
 [ 0.51681578 -0.99937487 -0.99572825  0.58383423]
 [ 0.97539145 -0.99992526  0.63370311 -0.99971783]
 [-0.1668677   0.9753989  -0.18295051 -0.9547317 ]
 [ 0.55140358 -0.99895197 -0.99290234  0.50107878]
 [ 0.96623731 -0.99973291 -0.93388748 -0.31682226]
 [-0.21054345  0.97485811 -0.20013551 -0.9611243 ]
 [ 0.99914366 -0.9999938   0.99965709 -0.9999792 ]
 [ 0.97247624 -0.99990141  0.71337819 -0.9997167 ]
 [-0.14636388  0.96758133 -0.22244608 -0.95604515]
 [ 0.99928123 -1.         -0.99716479 -0.99992251]
 [ 0.97743905 -0.99989218 -0.95567799 -0.52961254]
 [ 0.97534961 -0.99992567  0.61854506 -0.99970704]
 [ 0.96729648 -0.99974459 -0.93463451 -0.31141987]
 [ 0.95828623 -0.99862152  0.99489957 -0.99927521]
 [ 0.95441437 -0.9985069   0.99344522 -0.99913526]
 [ 0.97597235 -0.99992776  0.65945727 -0.99974716]
 [ 0.51761067 -0.99935633 -0.99562985  0.58312845]
 [ 0.99193919 -0.99998879  0.82787263 -0.99996555]
 [ 0.97526318 -0.99992543  0.61210358 -0.99970102]
 [ 0.55116415 -0.99902666 -0.9934848   0.50502086]
 [ 0.53999197  0.99860793  0.97644937  0.64463472]
 [-0.18897223  0.97842532 -0.17413273 -0.95799965]
 [-0.92434603  0.70981002 -0.978544    0.5789057 ]
 [ 0.35079247  0.99459082  0.54921305  0.59653437]
 [ 0.91332424 -0.99482864  0.99026763 -0.99627763]
 [ 0.72875607  0.98503101  0.95496762  0.73753572]
 [ 0.91351104 -0.99486792  0.99021769 -0.99630362]
 [ 0.91339803 -0.99484301  0.99025196 -0.99628723]
 [ 0.5774852  -0.99961585 -0.996562    0.60539067]
 [ 0.95532995 -0.99860495  0.9932577  -0.99920219]
 [ 0.99774998 -0.99997896  0.99922907 -0.99995691]
 [-0.84028518 -0.09277092 -0.98600495  0.54381204]
 [ 0.99915743 -0.99999368  0.99968636 -0.99997884]
 [ 0.99926931 -1.         -0.9968518  -0.99991357]
 [ 0.51121533 -0.99932969 -0.99555475  0.57909775]
 [ 0.99933684 -1.         -0.99742383 -0.99992752]
 [ 0.91359341 -0.99488664  0.99019092 -0.99631602]
 [ 0.955266   -0.99858493  0.99337417 -0.99919063]
 [ 0.91877389 -0.99529421  0.99120241 -0.99672651]
 [ 0.99904484 -0.99999446  0.999475   -0.99998343]
 [ 0.96276152 -0.99969071 -0.9310565  -0.32505652]]"""

h = x.split(" ")
print(h)
numbers = []
for i in range(0, len(h)):
    j = False
    try:
        number = float(h[i])
        numbers.append(number)
    except ValueError:
        j = True
    if j:
        try:
            number = float(h[i][:-2])
            numbers.append(number)
        except ValueError:
            pass
    if j:
        try:
            number = float(h[i][1:])
            numbers.append(number)
        except ValueError:
            try:
                number = float(h[i][2:])
                numbers.append(number)
            except ValueError:
                pass

arr = np.array(numbers)
arr = np.reshape(arr,(-1,4))




regularized = [arr[-1,:]]
for i in range(0, arr.shape[0] - 1):
    check = True
    for i2 in range(i+1, arr.shape[0]):
        dist = euclidean(arr[i,:],arr[i2,:])
        if dist < 0.0000001:
            print("shit")
            check = False
    if check:
        regularized.append(arr[i,:])

arr = np.array(regularized)
print(arr.shape[0])
pca = PCA(n_components=2)
Z = pca.fit_transform(arr)



pkl_file = open('encoder_input3.pkl', 'rb' )

data1 = pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('performance_list.pkl', 'rb')

data2 = pickle.load(pkl_file)
pkl_file.close()




datax_hidden, datax_hidden_t,\
datax_fce, datax_fce_t = data1[0], data1[1],\
                                                    data1[2],\
                                                            data1[3]
sanitized_list, for_dense_nn = [],[]
for i in range(200,datax_hidden_t.shape[0]):
    sanitized_list.append(sanitize_next_sample_for_gp([datax_hidden_t[i:i+1,:,:], datax_fce_t[i:i+1,:,:]],3,2,100,3))

    for_dense_nn.append(seriliaze_next_sample_for_loss_fce(sanitized_list[-1],3))


colors = ['blue', 'red']
color_list = []
for i in range(0,len(for_dense_nn)):

    color_list.append(colors[for_dense_nn[i][1]])

plt.scatter(Z[:,0], Z[:,1], c=color_list, s=0.1)
plt.show()



