import pickle

pkl_file = open('encoder_input.pkl', 'rb')
data2 = pickle.load(pkl_file)


pkl_file.close()
output = open('encoder_input2.pkl', 'wb')
pickle.dump(data2, output, protocol=2)
output.close()