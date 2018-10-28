from sklearn import tree

# we have oranges and apples. We know the weight of the fruit 
# and the surface of the fruit. A representation can be as below
#features = [[140,"smooth"], [130,"smooth"],[150,"bumpy"],[170,"bumpy"]]
#labels = ["apple","apple","orange","orange" ]

# In order to work with classifier, lets convert the strings into numbers

features = [[140,1], [130,1],[150,2],[170,2]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print(clf.predict([[140,1]]))


