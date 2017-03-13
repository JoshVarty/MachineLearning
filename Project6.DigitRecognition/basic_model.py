from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import sys

if sys.platform == 'win32': 
    data_root = 'C:\\data\\' # Change me to store data elsewhere
elif sys.platform == 'linux':
    data_root = '/home/jovarty/data'
else:
    raise Exception("Unknown OS")

pickle_file = 'five_digit_notMNIST.pickle'
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


image_height = 100
image_width = 100
num_labels = 10
output_size = 10;   #10 digits and one blank
lengthOfLabels = 5;
num_channels = 1; #grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_height, image_width, num_channels)).astype(np.float32)
  test = dataset.shape;
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  # Note that -1 is mapped to an all-zero vector
  newLabels = []
  for label in labels:
      newLabel = (np.arange(num_labels) == label[:,None]).astype(np.float32)
      #Flatten 5 vectors into one 
      newLabel = np.concatenate(newLabel).ravel()
      newLabels.append(newLabel)
      
  return dataset, np.array(newLabels)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("")
print("SHAPE AFTER REFORMAT")
print("")
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def ConvNet():
    batch_size = 32;
    patch_size = 4;
    depth_of_sixteen = 16;
    depth_of_sixty_four = 64;
    num_hidden = 64;

    fc_weight_size = 15680;

    graph = tf.Graph()
    with graph.as_default():
      # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 50))

      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      #Conv of 4x4x1x16
      layer1_weights = tf.get_variable("layer1_weights", [patch_size, patch_size, num_channels, depth_of_sixteen], initializer=tf.contrib.layers.xavier_initializer())
      layer1_biases = tf.Variable(tf.zeros([depth_of_sixteen]))
      #Conv of 4x4x16x16
      layer2_weights = tf.get_variable("layer2_weights", [patch_size, patch_size, depth_of_sixteen, depth_of_sixteen], initializer=tf.contrib.layers.xavier_initializer())
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_of_sixteen]))
      #Conv of 4x4x16x64
      layer3_weights = tf.get_variable("layer3_weights", [patch_size, patch_size, depth_of_sixteen, depth_of_sixty_four], initializer=tf.contrib.layers.xavier_initializer())
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth_of_sixty_four]))
      #Conv of 4x4x64x64
      layer4_weights = tf.get_variable("layer4_weights", [patch_size, patch_size, depth_of_sixty_four, depth_of_sixty_four], initializer=tf.contrib.layers.xavier_initializer())
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[depth_of_sixty_four]))

      #Output FC layers
      fc_1_weights = tf.get_variable("fc_1_weights", [fc_weight_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      fc_1_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))
      fc_2_weights = tf.get_variable("fc_2_weights", [fc_weight_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      fc_2_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))
      fc_3_weights = tf.get_variable("fc_3_weights", [fc_weight_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      fc_3_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))
      fc_4_weights = tf.get_variable("fc_4_weights", [fc_weight_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      fc_4_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))
      fc_5_weights = tf.get_variable("fc_5_weights", [fc_weight_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      fc_5_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))
      
      # Model
      def model(data):
        conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden_1 = tf.nn.relu(conv_1 + layer1_biases)

        conv_2 = tf.nn.conv2d(hidden_1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
 
        #TODO: Is this right?       
        pool_1 = tf.nn.max_pool(hidden_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv_3 = tf.nn.conv2d(pool_1, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden_3 = tf.nn.relu(conv_3 + layer3_biases)

        conv_4 = tf.nn.conv2d(hidden_3, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden_4 = tf.nn.relu(conv_4 + layer4_biases)

        pool_2 = tf.nn.max_pool(hidden_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        shape = pool_2.get_shape().as_list()
        reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])

        output_1 = tf.matmul(reshape, fc_1_weights) + fc_1_biases
        output_2 = tf.matmul(reshape, fc_2_weights) + fc_2_biases
        output_3 = tf.matmul(reshape, fc_3_weights) + fc_3_biases
        output_4 = tf.matmul(reshape, fc_4_weights) + fc_4_biases
        output_5 = tf.matmul(reshape, fc_5_weights) + fc_5_biases

        logits = tf.concat(1, [output_1, output_2, output_3, output_4, output_5])  

        return logits
      
      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
        
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))

      num_steps = 1001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        l = sum(l)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



ConvNet()