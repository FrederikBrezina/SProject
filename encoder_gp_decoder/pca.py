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
x = """[[-0.99670172 -0.97154826 -0.61060059 -0.73324925]
 [-0.99716836 -0.97665191 -0.61234581 -0.70476675]
 [-0.99987477 -0.99908715 -0.7655735  -0.20757496]
 [-0.99993378 -0.99967003 -0.74635375  0.16724636]
 [-0.99989319 -0.99950576 -0.71276397  0.12438681]
 [-0.99974072 -0.99822962 -0.72290421 -0.29860812]
 [-0.99999946 -0.99999666 -0.92637837  0.54666591]
 [-0.99990284 -0.99961156 -0.67504317  0.37455583]
 [-0.99995172 -0.99974662 -0.76893044  0.17648675]
 [-0.99993694 -0.99969405 -0.7451961   0.19296168]
 [-0.9999575  -0.99965727 -0.82201052 -0.10363595]
 [-0.99998999 -0.99992573 -0.8713131   0.07088602]
 [-0.99883085 -0.99460572 -0.56212598 -0.31083477]
 [-0.99993157 -0.99964267 -0.76384091  0.03017272]
 [-0.9999994  -0.99999571 -0.92923307  0.50929505]
 [-0.99888068 -0.99473739 -0.56903112 -0.31472892]
 [-0.97632498 -0.84958154 -0.40094993 -0.81503111]
 [-0.99983317 -0.99935597 -0.63584626  0.30808884]
 [-0.98394603 -0.85490263 -0.5140847  -0.86081618]
 [-0.99889016 -0.99156398 -0.65733027 -0.56871367]
 [-0.99996781 -0.99983382 -0.78271866  0.27208254]
 [-0.99055773 -0.94606429 -0.44722557 -0.70281708]
 [-0.99999464 -0.99997413 -0.830145    0.66214919]
 [-0.98305529 -0.84778744 -0.5091908  -0.86296856]
 [-0.99901712 -0.9953742  -0.5773806  -0.28710723]
 [-0.99930656 -0.99438375 -0.69546378 -0.54112399]
 [-0.99778569 -0.98568398 -0.58375412 -0.58597755]
 [-0.9999525  -0.99972004 -0.80274409 -0.07227907]
 [-0.99999398 -0.99997872 -0.76471233  0.85731715]
 [-0.99995935 -0.99965274 -0.82843018 -0.13031766]
 [-0.99999893 -0.99999404 -0.88557214  0.75538629]
 [-0.99999923 -0.9999972  -0.83104378  0.94532484]
 [-0.99448466 -0.96619278 -0.50576669 -0.67099065]
 [-0.99998987 -0.99992383 -0.87133163  0.06576747]
 [-0.99999803 -0.99998838 -0.88253212  0.61548924]
 [-0.99999613 -0.99997777 -0.86521322  0.52927238]
 [-0.9999761  -0.99976265 -0.86274862 -0.19521119]
 [-0.9999094  -0.99926734 -0.79292601 -0.23301448]
 [-0.99982542 -0.99905717 -0.71779931 -0.12821479]
 [-0.99999911 -0.99999541 -0.8805539   0.81838381]
 [-0.99994034 -0.9997406  -0.73484248  0.26903787]
 [-0.99960619 -0.99783957 -0.66466153 -0.22204781]
 [-0.99997842 -0.99988228 -0.81085676  0.25227571]
 [-0.99999976 -0.99999887 -0.93519306  0.67297041]
 [-0.99993068 -0.9996326  -0.7585023   0.07010599]
 [-0.99998599 -0.99989885 -0.85834914  0.04946341]
 [-0.99999487 -0.99997067 -0.85712683  0.49335447]
 [-0.99999428 -0.99996221 -0.87626636  0.26053366]
 [-0.99997634 -0.99980164 -0.84622085 -0.04312212]
 [-0.99998182 -0.99991065 -0.79642057  0.43058556]
 [-0.99998766 -0.99991572 -0.84737241  0.23144007]
 [-0.99995941 -0.99970859 -0.81600165 -0.05346381]
 [-0.99236488 -0.95803612 -0.4554756  -0.6668064 ]
 [-0.99997342 -0.99975532 -0.86254752 -0.26867425]
 [-0.99954307 -0.99727595 -0.66994309 -0.30334815]
 [-0.99991763 -0.99944925 -0.77546126 -0.09243514]
 [-0.99999762 -0.99998862 -0.85253751  0.75970107]
 [-0.99998963 -0.9999218  -0.87082231  0.0628062 ]
 [-0.97075921 -0.79030281 -0.41783366 -0.8561641 ]
 [-0.99996638 -0.99973667 -0.82676899 -0.03640997]
 [-0.99998635 -0.99994344 -0.80260944  0.47441673]
 [-0.99999964 -0.99999821 -0.9254064   0.70385444]
 [-0.99968338 -0.99847144 -0.65858471 -0.11172368]
 [-0.99844468 -0.99049091 -0.59765136 -0.51419079]
 [-0.99998546 -0.99985272 -0.87824601 -0.13658936]
 [-0.99965477 -0.99828774 -0.65577304 -0.13201565]
 [-0.99882853 -0.99365306 -0.59468639 -0.41108587]
 [-0.9999705  -0.99985075 -0.78536171  0.28717723]
 [-0.99895197 -0.99504316 -0.57506001 -0.3078357 ]
 [-0.99324375 -0.96247125 -0.46782336 -0.6571523 ]
 [-0.99975789 -0.99904543 -0.62234271  0.18876331]
 [-0.99991637 -0.99930573 -0.7987262  -0.23392406]
 [-0.99993485 -0.99969161 -0.73996127  0.20666198]
 [-0.99894613 -0.99510306 -0.570337   -0.29369214]
 [-0.99998915 -0.99992174 -0.86679667  0.08136117]
 [-0.99865735 -0.9927125  -0.58554316 -0.43310505]
 [-0.99961686 -0.99782461 -0.67275864 -0.24481221]
 [-0.9999941  -0.99996459 -0.86718339  0.33713272]
 [-0.99996036 -0.99958009 -0.84847081 -0.26984435]
 [-0.99987715 -0.99887669 -0.80273509 -0.42489636]
 [-0.99999475 -0.9999671  -0.86365217  0.43943399]
 [-0.93689156 -0.55901229 -0.37254569 -0.90511769]
 [-0.99990565 -0.99962205 -0.6769793   0.37836438]
 [-0.99998724 -0.99993575 -0.81256652  0.46816671]
 [-0.99991649 -0.99955195 -0.74650675  0.06229525]
 [-0.99999905 -0.99999434 -0.9001568   0.69089717]
 [-0.99999934 -0.99999756 -0.84484178  0.94093311]
 [-0.99999762 -0.99998808 -0.86587048  0.67024434]
 [-0.99999654 -0.99997014 -0.89024353  0.41115361]
 [-0.99999708 -0.99997556 -0.90245956  0.27456239]
 [-0.99943566 -0.99770331 -0.58464348 -0.07224516]
 [-0.99979937 -0.99919796 -0.63743657  0.21081805]
 [-0.99999613 -0.99997246 -0.89244872  0.25134477]
 [-0.99994707 -0.99955148 -0.81782901 -0.16480802]
 [-0.99995977 -0.99960524 -0.84296489 -0.23900056]
 [-0.99998379 -0.99992579 -0.78507137  0.54452896]
 [-0.99995953 -0.99982142 -0.74218261  0.41696608]
 [-0.99993992 -0.99959713 -0.78888571 -0.03609793]
 [-0.99910533 -0.9959619  -0.57443225 -0.23685896]
 [-0.99898058 -0.99584538 -0.5434643  -0.19091539]]"""

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
for i in range(datax_hidden_t.shape[0]):
    sanitized_list.append(sanitize_next_sample_for_gp([datax_hidden_t[i:i+1,:,:], datax_fce_t[i:i+1,:,:]],3,2,100,3))
    print(datax_fce_t[i,:,:])
    for_dense_nn.append(seriliaze_next_sample_for_loss_fce(sanitized_list[-1],3))

colors = ['blue', 'red']
color_list = []
for i in range(0,len(for_dense_nn)):

    color_list.append(colors[for_dense_nn[i][-1]])

plt.scatter(Z[:,0], Z[:,1], c=color_list)
plt.show()



