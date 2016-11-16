#!/usr/bin/python

""" this example borrows heavily from the example
    shown on the sklearn documentation:

    http://scikit-learn.org/stable/modules/cross_validation.html

"""

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
features = iris.data
labels = iris.target

### import the relevant code and make your train/test split
from sklearn import cross_validation

### name the output datasets features_train, features_test,
### labels_train, and labels_test
### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result
labels_train, labels_test, features_train, features_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0) 



###############################################################

clf = SVC(kernel="linear", C=1.)
clf.fit(labels_train, features_train)

print clf.score(labels_test, features_test)


##############################################################
def submitAcc():
    return clf.score(labels_test, features_test)
