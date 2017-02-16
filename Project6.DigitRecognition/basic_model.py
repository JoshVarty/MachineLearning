from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os

data_root = 'C:\\data\\' # Change me to store data elsewhere
pickle_file = 'five_digit_notMNIST.pickle'
dest_file_path = os.path.join(data_root, pickle_file)

with open(dest_file_path, 'rb') as f:
  save = pickle.load(f)
  #train_dataset = save['train_dataset']
  #train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  #del save  # hint to help gc free up memory
  #print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_height = 28
image_width = 140  
num_labels = 10
lengthOfLabels = 5;

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_height * image_width)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  newLabels = []
  for label in labels:
      test = (np.arange(num_labels) == label[:,None]).astype(np.float32)
      newLabels.append(test)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, newLabels


#train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)