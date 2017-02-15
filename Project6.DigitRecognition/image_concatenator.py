#This portion of the code simply 


from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os
import random

data_root = 'C:\\data\\' # Change me to store data elsewhere
pickle_file = 'notMNIST.pickle'
dest_file_path = os.path.join(data_root, pickle_file)

with open(dest_file_path, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


def generate(dataset):
    newDataSet = []
    numImages = len(dataset)
    minImageLength = 1
    maxImageLength = 5
    for image in dataset:
        newImage = np.zeros((28, 28 * 5))
        #How long will the new image be?
        imageLength = random.randint(minImageLength, maxImageLength)

        print(imageLength)
        for i in range(0, imageLength):
            randomImage = random.choice(dataset)
            #Splice randomeImage into new image
            newImage[:, (i) * 28 : (i+1) * 28] = randomImage[:,:]
            print("From Image: " + str(i))

        for i in range(imageLength, 5):
            blankImage = np.zeros((28,28))
            newImage[:, (i) * 28 : (i+1) * 28] = blankImage
            print("AppendZerios: " + str(i))


            




    

generate(test_dataset)

