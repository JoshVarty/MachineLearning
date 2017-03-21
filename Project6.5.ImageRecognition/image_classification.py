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
#tests.test_one_hot_encode(one_hot_encode)

#helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
#valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))




import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    batch_size = None
    tensor = tf.placeholder(tf.float32,shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), name='x')
    return tensor


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    batch_size = None
    return tf.placeholder(tf.float32, shape=(batch_size, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
#tests.test_nn_image_inputs(neural_net_image_input)
#tests.test_nn_label_inputs(neural_net_label_input)
#tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    `:param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    `:param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    `:param conv_strides: Stride 2-D Tuple for convolution
    `:param pool_ksize: kernal size 2-D Tuple for pool
    `:param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    input_shape = x_tensor.get_shape().as_list()
    input_depth = input_shape[3]
    weights = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], input_depth, conv_num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    conv = tf.nn.conv2d(x_tensor, weights, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    hidden = tf.nn.relu(conv + bias)
    result = tf.nn.max_pool(hidden, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding="SAME")
    return result


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_con_pool(conv2d_maxpool)



def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    dynamic_shape = tf.shape(x_tensor)
    static_shape = x_tensor.get_shape().as_list()

    return tf.reshape(x_tensor, [dynamic_shape[0], static_shape[1] * static_shape[2] * static_shape[3]])

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_flatten(flatten)



def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.1))
    biases = tf.Variable(tf.constant(1.0, shape=[num_outputs]))
    return tf.matmul(x_tensor, weights) + biases


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_fully_conn(fully_conn)



def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.1))
    biases = tf.Variable(tf.constant(1.0, shape=[num_outputs]))
    return tf.matmul(x_tensor, weights) + biases


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_output(output)








def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_1 = conv2d_maxpool(x, conv_num_outputs=8, conv_ksize=(4,4), conv_strides=(1,1), pool_ksize=(2,2), pool_strides=(1,1))
    conv_2 = conv2d_maxpool(conv_1, conv_num_outputs=16, conv_ksize=(4,4), conv_strides=(2,2), pool_ksize=(2,2), pool_strides=(2,2))
    conv_3 = conv2d_maxpool(conv_2, conv_num_outputs=32, conv_ksize=(4,4), conv_strides=(1,1), pool_ksize=(2,2), pool_strides=(1,1))

    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flattened = flatten(conv_3)
    

    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    num_outputs = 10
    fc = fully_conn(flattened, num_outputs)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(fc, num_outputs)
    
    
    # return output
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)