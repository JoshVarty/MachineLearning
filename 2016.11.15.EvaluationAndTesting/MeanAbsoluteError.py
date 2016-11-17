import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
labels_train, labels_test, features_train, features_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0) 


reg1 = DecisionTreeRegressor()
reg1.fit(labels_train, features_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(features_test, reg1.predict(labels_test)))

reg2 = LinearRegression()
reg2.fit(labels_train, features_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(features_test, reg2.predict(labels_test)))

results = {
 "Linear Regression": mae(features_test, reg2.predict(labels_test)),
 "Decision Tree": mae(features_test, reg1.predict(labels_test))
}