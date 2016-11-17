# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('C:\\Users\\Josh\\Documents\\GitHub\\MachineLearning\\Project1.TitanicSurvivalExploration\\titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
labels_train, labels_test, features_train, features_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0) 

clf1 = DecisionTreeClassifier()
clf1.fit(labels_train, features_train)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(features_test, clf1.predict(labels_test)), precision(features_test, clf1.predict(labels_test)))

clf2 = GaussianNB()
clf2.fit(labels_train, features_train)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(features_test ,clf2.predict(labels_test)), precision(features_test, clf2.predict(labels_test)))

results = {
  "Naive Bayes Recall": recall(features_test ,clf2.predict(labels_test)),
  "Naive Bayes Precision": precision(features_test, clf2.predict(labels_test)),
  "Decision Tree Recall": recall(features_test, clf1.predict(labels_test)),
  "Decision Tree Precision": precision(features_test, clf1.predict(labels_test)))
}