#Assignment 1

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline


#First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. 
#The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. 
#Given these sizes, it should be possible to train models quickly on any machine.

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = 'C:\\data\\' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


#Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#Problem 1
#Let's take a peek at some of the data to make sure it looks sensible. 
#Each exemplar should be an image of a character A through J rendered in a different font. 
#Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.

from IPython.display import Image
import random
#Select random folder
folder = random.choice(test_folders)
files = os.listdir(folder)

for i in range(0,3):
    image_name = random.choice(files)
    image_path = os.path.join(folder, image_name)
    image = Image(filename=image_path)
    print(image_path)
    display(image)




#Now let's load the data in a more manageable format. Since, you might not be able to fit it all in memory, 
#we'll load each class into a separate dataset, store them on disk and curate them independently. 
#Later we'll merge them into a single dataset of manageable size.

#We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, 
#normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.

#A few images might not be readable, we'll just skip them.

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


#Problem 2
#Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. 
#Hint: you can use matplotlib.pyplot.

pickle_file = train_datasets[0]  # index 0 should be all As, 1 = all Bs, etc.
with open(pickle_file, 'rb') as f:
    letter_set = pickle.load(f)  # unpickle
    sample_idx = np.random.randint(len(letter_set))  # pick a random image index
    sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it



#Problem 3
#Another check: we expect the data to be balanced across classes. Verify that.
maxsize = -sys.maxsize - 1
minsize = sys.maxsize;
for i in range(0, len(train_datasets)):
    pickle_file = train_datasets[i]
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        length = len(letter_set)
        if length > maxsize:
            maxsize = length
        if length < minsize:
            minsize = length
        print(pickle_file)
        print("Length " + str(length))

print("Min: " + str(minsize))
print("Max: " + str(maxsize))
print("Maximum difference in class size: " + str(maxsize - minsize))