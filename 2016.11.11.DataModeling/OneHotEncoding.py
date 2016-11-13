# In this exercise we'll load the titanic data (from Project 0)
# And then perform one-hot encoding on the feature names

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to categorical data
X = X.select_dtypes(include=[object])

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#  Create a LabelEncoder object, which will turn all labels present in
#  in each feature to numbers. For example, the labels ['cat', 'dog', 'fish']
#  might be transformed into [0, 1, 2]

le = LabelEncoder()

# For each feature in X, apply the LabelEncoder's fit_transform
# function, which will first learn the labels for the feature (fit)
# and then change the labels to numbers (transform). 


for feature in X:
    josh = le.fit_transform(X[feature])
    X[feature] = josh

# Create a OneHotEncoder object, which will create a feature for each
# label present in the data. For example, for a feature 'animal' that had
# the labels ['cat','dog','fish'], the new features (instead of 'animal') 
# could be ['animal_cat', 'animal_dog', 'animal_fish']

ohe = OneHotEncoder() 

# Apply the OneHotEncoder's fit_transform function to all of X, which will
# first learn of all the (now numerical) labels in the data (fit), and then
# change the data to one-hot encoded entries (transform).
onehotlabels = ohe.fit_transform(X)

