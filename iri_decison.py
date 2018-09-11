from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris=load_iris()
removed=[2,54,105]
new_target=np.delete(iris.target,removed)
new_data=np.delete(iris.data,removed,axis=0)

clf=tree.DecisionTreeClassifier()

clf=clf.fit(new_data,new_target)

prediction=clf.predict(iris.data[removed])
print ("labels predicted ",prediction)

