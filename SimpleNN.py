import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("new.txt", delimiter=" ")
# split into input (X) and output (Y) variables
X = dataset[:,0:2]
print(X)

Y = dataset[:,2]
print(Y)