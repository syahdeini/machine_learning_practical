# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import pdb


experiment_name = "2_layer_conv"

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


def _variable_with_weight_decay(shape, mean,stddev):
    initial = tf.truncated_normal(shape = shape, mean =mean, stddev=stddev)
    return tf.Variable(initial)

num_hidden2 = 100
num_hidden3 = 50
BATCH_SIZE = 40
NUM_CLASSES = 10
train_data = CIFAR10DataProvider('train', batch_size=BATCH_SIZE)
valid_data = CIFAR10DataProvider('valid', batch_size=BATCH_SIZE)
train_data.inputs = train_data.inputs.reshape((-1, 1024, 3), order='F')
train_data.inputs = train_data.inputs.reshape((-1, 32, 32, 3))
valid_data.inputs = valid_data.inputs.reshape((-1, 1024, 3), order='F')
valid_data.inputs = valid_data.inputs.reshape((-1, 32, 32, 3))


input_dim = 32
output_dim = 32
# place holder for input and target
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1], train_data.inputs.shape[2], train_data.inputs.shape[3]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
# pdb.set_trace()
# building graph
stddev = tf.placeholder("float")
mean = tf.placeholder("float")
lrate = tf.placeholder("float")

conv1_out_size = 14 #number of output channel of first convolutional 
with tf.name_scope('conv-1') as scope:
    kernel = _variable_with_weight_decay(
                                         shape=[5, 5, 3, conv1_out_size],
                                         stddev=stddev,
                                         mean=mean)

    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation = tf.nn.bias_add(conv, biases)
    # conv1 = tf.nn.relu(pre_activation)
    local1 = tf.nn.relu(pre_activation)
    # pool1
    pool1 = tf.nn.max_pool(local1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')

with tf.name_scope('conv-2') as scope:
#    pdb.set_trace()
    kernel2 = _variable_with_weight_decay(
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=stddev,
                                         mean=mean)

    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
    local2 = tf.nn.relu(pre_activation)
    # pool1
    pool2 = tf.nn.max_pool(local2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
    
# Move everything into depth so we can perform a single matrix multiply.
with tf.name_scope('Dense-Relu_Layer') as scope:
    # flattening the input
    last_layer = pool2
    tot_shape=last_layer.get_shape()[1].value*last_layer.get_shape()[2].value*last_layer.get_shape()[3].value
    reshape = tf.reshape(last_layer, [BATCH_SIZE,tot_shape])
    weights = _variable_with_weight_decay('weights3', shape=[tot_shape, tot_shape],stddev=1.0, wd=0.0)
    biases = tf.Variable(tf.zeros([tot_shape]), 'biases') 
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)


with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights5', [tot_shape, NUM_CLASSES],
                                          stddev=1.0, wd=0.0)
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), 'biases') 
    softmax_linear = tf.add(tf.matmul(local3, weights), biases)
    soft_max_out = tf.nn.softmax(softmax_linear)

with tf.name_scope('error'):
    out_login = tf.nn.softmax_cross_entropy_with_logits(logits = softmax_linear, labels=targets)
    error = tf.reduce_mean(out_login)

# use softmax for accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(softmax_linear, 1), tf.argmax(targets, 1)), 
            tf.float32))

# use adam optimizer 
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(error)
    
init = tf.global_variables_initializer()
# begin training


slr = np.linspace(0.0001,0.005,3)
smean = np.linspace(0,0.5,3)
sstd = np.linspace(0.1,1.0,3)


for _lr in slr:
    for _mean in smean:
        for _std in sstd: 
  
