#This portion of the code simply 

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import os
import sys
import random
from PIL import Image

if sys.platform == 'win32': 
    data_root = 'C:\\data\\' # Change me to store data elsewhere
elif sys.platform == 'linux':
    data_root = '/home/jovarty/data'
else:
    raise Exception("Unknown OS")

pickle_file = 'notMNIST.pickle'
dest_file_path = os.path.join(data_root, pickle_file)

with open(dest_file_path, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset'][:100000]
  train_labels = save['train_labels'][:100000]
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


def generate(dataset, labels):
    newDataSet = []
    newLabels = []
    numImages = len(dataset)
    minImageLength = 1
    maxImageLength = 5
    for index, image in enumerate(dataset):
        imageArray = np.zeros((28, 28 * 5))
        newLabel = []
        #How long will the new image be?
        imageLength = random.randint(minImageLength, maxImageLength)

        #print(imageLength)
        for i in range(0, imageLength):
            randomInt = random.randint(0, len(dataset) - 1)
            randomImage = dataset[randomInt]
            #NOTE: We're adding one for all labels to account for blank images which we'll classify as 0
            newLabel.append(labels[randomInt] + 1)
            #Splice randomeImage into new image
            imageArray[:, (i) * 28 : (i+1) * 28] = randomImage[:,:]
            #print("From Image: " + str(i))

        for i in range(imageLength, 5):
            blankImage = np.zeros((28,28))
            blankImage.fill(0)
            #NOTE: We're representing blanks as 0 
            newLabel.append(0)
            imageArray[:, (i) * 28 : (i+1) * 28] = blankImage
            #print("AppendZerios: " + str(i))

        imageHeight = 40
        imageWidth = 40
        image = Image.fromarray(imageArray)
        squareImage = image.resize([imageHeight, imageWidth])
        data = np.array(squareImage.getdata())
        data.resize((imageHeight, imageWidth))
        
        newLabels.append(newLabel)
        newDataSet.append(data)

        if index % 1000 == 0:
            print("Index: " + str(index));
            #plt.figure()
            #plt.imshow(data)
            #plt.imshow(data)
            #print(newLabel)

    return np.array(newDataSet), np.array(newLabels)

newTrainData, newTrainLabels = generate(train_dataset, train_labels)
print("Loaded training data")
newValidData, newValidLabels = generate(valid_dataset, valid_labels)
print("Loaded validation data")
newTestData, newTestLabels = generate(test_dataset, test_labels)
print("Loaded testing data")

print("loaded all");

del train_dataset
del train_labels
del valid_dataset
del valid_labels
del test_dataset
del test_labels

#Finally, let's save the data for later reuse:
pickle_file = os.path.join(data_root, 'five_digit_notMNIST.pickle')

print("Pickle_file: " + str(pickle_file));

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': newTrainData,
    'train_labels': newTrainLabels,
    'valid_dataset': newValidData,
    'valid_labels': newValidLabels,
    'test_dataset': newTestData,
    'test_labels': newTestLabels,
    }
  print("dumping");
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  print("dumped");
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

print("Finished writing to disk")

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
