'''Temperature converter using error correction - the program is given a ground truth value in the Celsius scale for a temperature in Fahrenheit. After that the user inputs
a temperature in the Fahrenheit scale. The program tries to predict the value in Celsius. The error is calculated based on the ground truth value. The error correction is then
applied to correct the prediction.

Since the mapping from Celsius to Fahrenheit is linear, i.e. of the form C = mF where m is some real value, we try to randomly predict m. Based on the error (difference between
the ground truth and the predicted output in the training data, we calculate the error in m that leads to the error in the predicted output. We then apply the error correction to
m and run it for out test data.'''

import numpy as np

print("Enter a temperature in Fahrenheit scale to convert to Celsius scale.")
f_test = float(input()) # user input test data
f_train = 200 * np.random.random((50, 1)) + 100 # randomly generates a 50 x 1 matrix of traning values in the range [100, 300]
c_gt = np.array(0.55 * f_train) # - 17.78 # ground truth matrix of centigrade values generated from the training set using the conversion formula C = (5 * (F - 32)/9)
m = 100 * np.random.random() # random scaling factor - paramter to be learned
# print ("guess")
# print (m)
c_out = m * f_train
#for i in range(500):
error = np.array(c_gt - c_out)
delta_m = error / f_train
# for x in np.nditer(delta_m):
m += delta_m[0]
'''print("training set")
print (f_train)
print("prediction")
print (c_out)
print("ground truth")
print (c_gt)
print("error")
print (error)
print ("guess error")
print (delta_m)'''
#result = m * f_test
print ((m * f_test) - 17.78)     
#c_test = int(m * f_test) # - 17.78
#print (c_test)