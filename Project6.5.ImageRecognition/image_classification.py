"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir, join
from tqdm import tqdm
import problem_unittests as tests
import tarfile
from sklearn.preprocessing import LabelBinarizer

cifar10_dataset_folder_path = 'cifar-10-batches-py'
cifar10_dataset_file_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(cifar10_dataset_file_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(cifar10_dataset_file_path) as tar:
        tar.extractall()
        tar.close()

tests.test_folder_path(cifar10_dataset_folder_path)



#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import helper
import numpy as np

# Explore the dataset
batch_id = 1
sample_id = 5
#helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)



def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """

    newImages = np.empty_like(x)
    newImages = x / 255
    return np.array(newImages)



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)


encoder = LabelBinarizer()
encoder.fit([0,1,2,3,4,5,6,7,8,9])
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return encoder.transform(x)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)

helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


