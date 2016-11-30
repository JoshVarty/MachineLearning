import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


### your code goes here--should return a trained decision tree classifer

clf = DecisionTreeClassifier(min_samples_split = 2)
clf.fit(features_train, labels_train)

acc_min_samples_split_2 = clf.score(features_test, labels_test)

clf = DecisionTreeClassifier(min_samples_split = 50)
clf.fit(features_train, labels_train)

acc_min_samples_split_50 = clf.score(features_test, labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}