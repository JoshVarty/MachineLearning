import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


#################################################################################


########################## DECISION TREE #################################



#### your code goes here

from sklearn.tree import DecisionTreeClassifier
### your code goes here--should return a trained decision tree classifer

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)


acc = clf.score(features_test, labels_test)

def submitAccuracies():
  return {"acc":round(acc,3)}

