from sklearn import tree

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47],
	 [175, 64, 39], [177, 70, 40], [159, 68, 41], [152, 52, 36], [171, 75, 42], [181, 85, 43]]
#labels
Y = ['male', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male']

#create decision tree classifer
clf = tree.DecisionTreeClassifier()

#train the model using our datasets as parameters
clf = clf.fit(X, Y)

#test prediciton
prediction = clf.predict([[125, 50, 38]])

#display the prediciton
print (prediction)