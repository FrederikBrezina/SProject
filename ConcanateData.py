dataset = np.loadtxt("units_of_first_layer.txt", delimiter=" ")
X = dataset[:, 0:1]
Y = dataset[:, 1:2]
dataset = np.loadtxt("units_of_first_layer_from_100_to_390.txt", delimiter=" ")
X1 = dataset[:, 0:1]
Y1 = dataset[:, 1:2]
print(X)
dataset = np.loadtxt("units_of_first_layer_from_420_to_3000.txt", delimiter=" ")
X2 = dataset[:, 0:1].tolist()
Y2 = dataset[:, 1:2].tolist()
x = []
y = []
for i in range(0, len(X)):
    x.append(X[i])
    y.append(Y[i])
for i in range(0, len(X1)):
    x.append(X1[i])
    y.append(Y1[i])
for i in range(0, len(X2)):
    x.append(X2[i])
    y.append(Y2[i])
plt.figure(1)
plt.plot(x,y)
plt.show()