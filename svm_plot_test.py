import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

X = np.array([[-2, 4, -1],
			  [4, 1, -1],
			  [1, 6, -1],
			  [2, 4, -1],
			  [6, 2, -1]])
			  
Y = np.array([-1, -1, 1, 1, 1])

for d, sample in enumerate(X):
	if d < 2:
		plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
	else:
		plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

plt.plot([-2, 6], [6, 0.5]) # plots a line
plt.xlabel("Gagan")
plt.ylabel("Akash")
plt.show()