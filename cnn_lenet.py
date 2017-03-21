# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
#import matplotlib.pyplot as plt
import pdb

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
       'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def list_to_file(thelist,filename):
    thefile = open(filename, 'w')
    for item in thelist:
      thefile.write("%s\n" % item)


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


BATCH_SIZE = 100
NUM_CLASSES = 10
train_data = CIFAR10DataProvider('train', batch_size=BATCH_SIZE)
valid_data = CIFAR10DataProvider('valid', batch_size=BATCH_SIZE)
train_data.inputs = train_data.inputs.reshape((-1, 1024, 3), order='F')
train_data.inputs = train_data.inputs.reshape((-1, 32, 32, 3))
valid_data.inputs = valid_data.inputs.reshape((-1, 1024, 3), order='F')
valid_data.inputs = valid_data.inputs.reshape((-1, 32, 32, 3))

# place holder for input and target
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1], train_data.inputs.shape[2], train_data.inputs.shape[3]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
# building graph
with tf.name_scope('conv-1') as scope:
    n_actv_map = 6
    kernel = _variable_with_weight_decay('weights-conv2',
                                         shape=[5, 5, 3,n_actv_map],
                                         stddev=5e-2,
                                         wd=0.0)

    conv = tf.nn.conv2d(inputs, kernel, [1, 1,1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = tf.Variable(tf.zeros([n_actv_map]), 'biases') 
    pre_activation = tf.nn.bias_add(conv, biases)
    # conv1 = tf.nn.relu(pre_activation)
    local1 = tf.nn.relu(pre_activation)
    # pool1
    pool1 = tf.nn.max_pool(local1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME')

with tf.name_scope('conv-2') as scope:
    n_actv_map = 16
    kernel2 = _variable_with_weight_decay('weights-conv2',
                                         shape=[5, 5, 6,n_actv_map],
                                         stddev=5e-2,
                                         wd=0.0)

    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([n_actv_map]), 'biases') 
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
    local2 = tf.nn.relu(pre_activation2)
    # pool1
    pool2 = tf.nn.max_pool(local2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME')

with tf.name_scope('normalReluLayer') as scope:
    in_dim = 16
    # Move everything into depth so we can perform a single matrix multiply.
    shape_pool = pool2.get_shape().as_list()
    out_dim = shape_pool[1]*shape_pool[2]*shape_pool[3]
  #  pdb.set_trace()
    reshapeN1 = tf.reshape(pool2, [BATCH_SIZE,out_dim])
   # dim = reshape.get_shape()[1].value
    weightsN1 = _variable_with_weight_decay('weights3', shape=[out_dim,120],
                                          stddev=0.04, wd=0.0)
    biasesN1 = tf.Variable(tf.zeros([120]), 'biases') 
    localN1 = tf.nn.relu(tf.matmul(reshapeN1, weightsN1) + biasesN1)

with tf.name_scope('normalReluLayer2') as scope:
    in_dim = 120
    weightsN2 = _variable_with_weight_decay('weights3', shape=[in_dim,84],
                                          stddev=0.04, wd=0.0)
    biasesN2 = tf.Variable(tf.zeros([84]), 'biases') 
    localN2 = tf.nn.relu(tf.matmul(localN1, weightsN2) + biasesN2)

with tf.name_scope('normalReluLayer2') as scope:
    in_dim = 84
    weightsN3 = _variable_with_weight_decay('weights3', shape=[in_dim,10],
                                          stddev=0.04, wd=0.0)
    biasesN3 = tf.Variable(tf.zeros([10]), 'biases') 
    localN3 = tf.nn.relu(tf.matmul(localN2, weightsN3) + biasesN3)

with tf.variable_scope('softmax_linear') as scope:
    out_dim = 10
    weights = _variable_with_weight_decay('weights5', [out_dim, NUM_CLASSES],
                                          stddev=0.05, wd=0.0)
    # biases = _variable_on_cpu('biases5', [NUM_CLASSES],
    #                           tf.constant_initializer(0.0))
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), 'biases') 
    softmax_linear = tf.add(tf.matmul(localN3, weights), biases)

    soft_max_out = tf.nn.softmax(softmax_linear)

with tf.name_scope('error'):
    out_login = tf.nn.softmax_cross_entropy_with_logits(logits = softmax_linear, labels=targets)
    error = tf.reduce_mean(out_login)


with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(softmax_linear, 1), tf.argmax(targets, 1)), 
            tf.float32))
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(error)
    
init = tf.global_variables_initializer()
# begin training

acc_train_list = []
err_train_list = []
acc_valids = []
err_valids = []
with tf.Session() as sess:
    sess.run(init)
    for e in range(120):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:            
            # running sesssion
            # input_batch=tf.reshape(input_batch,[BATCH_SIZE,32,32,3])
            _our, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            # calculating error and accuracy for batch
	    running_error += batch_error
            running_accuracy += batch_acc
#            print("batch acc"+str(batch_acc))

        # averaging the error and accuracy
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
        acc_train_list.append(running_accuracy)
        err_train_list.append(running_error)

        # validation
        if  (e + 1) % 5 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                #input_batch=tf.reshape(input_batch,[BATCH_SIZE,32,32,3])
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
                #print("batch acc"+str(batch_acc))
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
            acc_valids.append(valid_accuracy)
            err_valids.append(valid_error)

list_to_file(err_train_list,"lenet_error_trains_stanford.txt")
list_to_file(acc_train_list,"lenet_acc_trains_stanford.txt")
list_to_file(err_valids,"lenet_error_valid_stanford.txt")
list_to_file(acc_valids,"lenet_acc_valid_stanford.txt")
