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
x = """[[-0.17328556  0.03724857  0.86543721  0.89194143 -0.38020855  0.69650209]
 [-0.41408426 -0.13357867  0.87394345  0.91880882 -0.33624604  0.69806349]
 [ 0.10087604  0.21259923  0.856022    0.85521722 -0.42420119  0.69487011]
 [ 0.1227675   0.22624281  0.8552407   0.85181671 -0.42764983  0.69473886]
 [-0.75885975 -0.45574096  0.89007419  0.95569849 -0.23999436  0.70129466]
 [-0.99999827 -0.99969685  0.97544295  0.9999457   0.71713209  0.72781169]
 [ 0.10020997  0.21218383  0.85604566  0.85531926 -0.42409632  0.69487411]
 [-0.52262259 -0.22082071  0.87820697  0.93022889 -0.3125433   0.69888079]
 [-0.37152851 -0.10148633  0.87237012  0.91426092 -0.34470248  0.69776797]
 [-0.99997562 -0.99892694  0.9749431   0.9997015   0.75828892  0.44386137]
 [-1.         -0.99997771  0.98613501  0.99999493  0.85818791  0.74520326]
 [-0.267378   -0.02672799  0.86866939  0.90282029 -0.36400333  0.69708502]
 [-1.         -0.99998015  0.9934392   0.99999511  0.93778014  0.41988418]
 [-0.99948192 -0.9919439   0.9396497   0.99838102  0.39027706  0.67201579]
 [-0.39828339 -0.12154263  0.87335408  0.91712677 -0.3394317   0.69795239]
 [-0.22158934  0.00476008  0.86708796  0.89760321 -0.37200624  0.69679826]
 [-0.99999875 -0.99974692  0.97663754  0.999955    0.72996414  0.73063624]
 [-0.40704632 -0.1281994   0.87368011  0.91806048 -0.337672    0.69801378]
 [-1.         -0.99999785  0.99494714  0.99999934  0.95853531  0.55226982]
 [-0.99997818 -0.99870968  0.96320385  0.99975413  0.59510869  0.70329952]
 [-1.         -0.99998784  0.98773408  0.99999702  0.87994319  0.74758005]
 [-0.9961794  -0.97469926  0.91765445  0.99467444  0.2356922   0.65035111]
 [-0.76108164 -0.45843112  0.8902179   0.95595515 -0.23905219  0.7013253 ]
 [-0.80489254 -0.51444387  0.89327741  0.96115661 -0.21860085  0.7019853 ]
 [-0.99999928 -0.99982119  0.97877437  0.99996853  0.75324529  0.73599494]
 [-0.99997687 -0.99896002  0.97529685  0.9997105   0.76175708  0.44100767]
 [-1.         -0.99999875  0.99226069  0.99999958  0.9366191   0.75632906]
 [-1.         -0.99998921  0.98802447  0.99999732  0.88381314  0.74804235]
 [-0.99999601 -0.99962598  0.98254555  0.99989843  0.83090937  0.42981279]
 [-0.92228681 -0.86151385  0.86882067  0.96760386 -0.02203796  0.6151554 ]
 [-0.27529567 -0.03224929  0.86894506  0.90370929 -0.36259347  0.69713533]
 [-0.9999997  -0.9999066   0.98512286  0.9999792   0.84464121  0.62079394]
 [-0.15328753  0.05050739  0.86475712  0.88954359 -0.38354367  0.69638097]
 [-0.62479722 -0.31255469  0.88272762  0.94097197 -0.2860938   0.69977504]
 [-0.92228681 -0.86151385  0.86882067  0.96760386 -0.02203796  0.6151554 ]
 [-0.61885142 -0.30688798  0.88244569  0.94034123 -0.28778437  0.69971836]
 [-0.62366152 -0.31146875  0.88267356  0.94085139 -0.28641844  0.69976419]
 [-0.99993289 -0.99802953  0.9666754   0.99946266  0.67847574  0.50402617]
 [-0.74264759 -0.43649608  0.889054    0.9538424  -0.24663855  0.70107841]
 [-0.18309283  0.03070718  0.86577129  0.89310539 -0.37856078  0.69656181]
 [-0.99999833 -0.99974561  0.97794664  0.9999429   0.76882672  0.6539638 ]
 [-0.23095286 -0.00162071  0.86740977  0.89868116 -0.37038925  0.69685638]
 [-0.99999994 -0.99996942  0.98918921  0.99999332  0.88830894  0.63373673]
 [-0.99999374 -0.99952197  0.98220229  0.99986577  0.82973754  0.38636753]
 [-1.         -0.9999969   0.99436051  0.99999911  0.95213324  0.56672943]
 [-0.58528662 -0.27572441  0.88090283  0.93679827 -0.29693863  0.6994105 ]
 [-0.77057052 -0.47007117  0.89084274  0.95705843 -0.23493592  0.70145881]
 [-0.50123101 -0.20291762  0.87733155  0.92798913 -0.31750596  0.69871098]
 [-0.16811216  0.04068866  0.86526114  0.89132434 -0.38107446  0.69647068]
 [-0.99842739 -0.98476791  0.92824614  0.99686211  0.30644789  0.66011655]]"""

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
arr = np.reshape(arr,(-1,6))




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

plt.scatter(Z[:,0], Z[:,1], c=color_list, s=0.4)
plt.show()



