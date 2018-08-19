import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]) # input 
y = np.array([[0,1,1,0]]).T # transpose of the ground truth vector

syn0 = 2*np.random.random((3,4)) - 1 # create a 3x4 weight matrix of random values in the range [-1,1]
syn1 = 2*np.random.random((4,1)) - 1 # create a 4x1 weight matrix of random values in the range [-1,1]


print("Input Data\n")
print(X)
print("\n")
print("Ground Truth\n")
print(y)
print("\n")

print("First weight matrix\n")
print(syn0)
print("\n")
print("Second weight matrix\n")
print(syn1)

for j in range(60000):
	layer_one = 1/(1+np.exp(-(np.dot(X,syn0)))) # sigmoid activation
	layer_two = 1/(1+np.exp(-(np.dot(layer_one,syn1))))
	layer_two_delta = (y - layer_two)*(layer_two*(1-layer_two)) # gradient of the sigmoid function = X * (1 - X)
	layer_one_delta = layer_two_delta.dot(syn1.T) * (layer_one * (1-layer_one))
	syn1 += layer_one.T.dot(layer_two_delta)
	syn0 += X.T.dot(layer_one_delta)
	
print("\nOutput:")
print("Layer one loss\n")
print(layer_one_delta)
print("\n")
print("Layer two loss\n")
print(layer_two_delta)
print("\n")

print("Layer one\n")	
print(layer_one)
print("\n")
print("Layer two - predicted output\n")
print(layer_two)