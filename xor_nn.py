import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):
	layer_one = 1/(1+np.exp(-(np.dot(X,syn0))))
	layer_two = 1/(1+np.exp(-(np.dot(layer_one,syn1))))
	layer_two_delta = (y - layer_two)*(layer_two*(1-layer_two))
	layer_one_delta = layer_two_delta.dot(syn1.T) * (layer_one * (1-layer_one))
	syn1 += layer_one.T.dot(layer_two_delta)
	syn0 += X.T.dot(layer_one_delta)
	
print(layer_two)