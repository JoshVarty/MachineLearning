# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os

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

  

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


#Introduce and tune L2 regularization for both logistic and neural network models. 
#Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. 
#In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). 
#The right amount of regularization should improve your validation / test accuracy.

def LogisticRegression(l2_weight):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      
      # Variables.
      weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))
      
      # Training computation.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + l2_weight * tf.nn.l2_loss(weights)
      
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


      num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

      return accuracy(test_prediction.eval(), test_labels)

#result = LogisticRegression(0.0);
#print("0.000 " + str(result) + "%"); # 86.21%
#result = LogisticRegression(0.001);
#print("0.001 " + str(result) + "%"); # 89.16%


def OneHiddenLayer(l2_weight):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      hidden_layer_size = 1024
      hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
      hidden_biases = tf.Variable(tf.zeros([hidden_layer_size]))
      hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

      # Variables.
      weights = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))
      
      # Training computation.
      logits = tf.matmul(hidden_layer, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 
      + l2_weight * tf.nn.l2_loss(weights) 
      + l2_weight * tf.nn.l2_loss(hidden_weights)
      
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)

      valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
      valid_logits = tf.matmul(valid_hidden, weights) + biases
      valid_prediction = tf.nn.softmax(valid_logits)

      test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
      test_logits = tf.matmul(test_hidden, weights) + biases
      test_prediction = tf.nn.softmax(test_logits)


    num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #if (step % 500 == 0):
          #print("Minibatch loss at step %d: %f" % (step, l))
          #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

      return accuracy(test_prediction.eval(), test_labels)

#result = OneHiddenLayer(0.0);
#print("0.000 " + str(result) + "%"); 
#result = OneHiddenLayer(0.001);
#print("0.001 " + str(result) + "%"); 
#result = OneHiddenLayer(0.010);
#print("0.010 " + str(result) + "%"); 
#result = OneHiddenLayer(0.100);
#print("0.100 " + str(result) + "%"); 
#result = OneHiddenLayer(0.5);
#print("0.500 " + str(result) + "%"); 



#Problem 2
#Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

def OneHiddenLayer_WithSmallBatchSize(l2_weight):
    batch_size = 3

    graph = tf.Graph()
    with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      hidden_layer_size = 1024
      hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
      hidden_biases = tf.Variable(tf.zeros([hidden_layer_size]))
      hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

      # Variables.
      weights = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))
      
      # Training computation.
      logits = tf.matmul(hidden_layer, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 
      + l2_weight * tf.nn.l2_loss(weights) 
      + l2_weight * tf.nn.l2_loss(hidden_weights)
      
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)

      valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
      valid_logits = tf.matmul(valid_hidden, weights) + biases
      valid_prediction = tf.nn.softmax(valid_logits)

      test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
      test_logits = tf.matmul(test_hidden, weights) + biases
      test_prediction = tf.nn.softmax(test_logits)


    num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

      return accuracy(test_prediction.eval(), test_labels)

result = OneHiddenLayer_WithSmallBatchSize(0.0)
print("0.000 " + str(result) + "%");



# Problem 3
# Introduce Dropout on the hidden layer of the neural network. 
# Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. 
# TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
# What happens to our extreme overfitting case?

def OneHiddenLayer_Dropout(l2_weight):
    batch_size = 10     #Note that even with a small batch size, if we use dropout the results seem to generalize and score us ~88%.
                        #I also had to lower the learning rate, though.

    graph = tf.Graph()
    with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      hidden_layer_size = 1024
      hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
      hidden_biases = tf.Variable(tf.zeros([hidden_layer_size]))
      hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

      keep_prob = tf.placeholder(tf.float32)
      hidden_layer_drop = tf.nn.dropout(hidden_layer, keep_prob)

      # Variables.
      weights = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))
      
      # Training computation.
      logits = tf.matmul(hidden_layer_drop, weights) + biases
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 
      + l2_weight * tf.nn.l2_loss(weights) 
      + l2_weight * tf.nn.l2_loss(hidden_weights)
      
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)

      valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
      valid_logits = tf.matmul(valid_hidden, weights) + biases
      valid_prediction = tf.nn.softmax(valid_logits)

      test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
      test_logits = tf.matmul(test_hidden, weights) + biases
      test_prediction = tf.nn.softmax(test_logits)


    num_steps = 5001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run(feed_dict = { keep_prob : 0.5 })
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 1.0}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

      return accuracy(test_prediction.eval(), test_labels)


#result = OneHiddenLayer_Dropout(0.0)
#print("Dropout: " + str(result) + "%");


