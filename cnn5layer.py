# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import pdb


experiment_name = "5_layer_conv"

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
  dtype = tf.float32
  var =  tf.Variable(
        tf.truncated_normal(
            shape), 
        'weights')

  if wd > 0:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


num_hidden = 200
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

conv1_out_size = 14 #number of output channel of first convolutional 
with tf.name_scope('conv-1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

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
    kernel2 = _variable_with_weight_decay('weights2',
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
    local2 = tf.nn.relu(pre_activation2)
    # pool1
    pool2 = tf.nn.max_pool(local2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
with tf.name_scope('conv-3') as scope:
#    pdb.set_trace()
    kernel3 = _variable_with_weight_decay('weights3',
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases3 = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation3 = tf.nn.bias_add(conv3, biases3)
    # conv1 = tf.nn.relu(pre_activation)
    local3 = tf.nn.relu(pre_activation3)
    # pool1
    pool3 = tf.nn.max_pool(local3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
    
with tf.name_scope('conv-4') as scope:
#    pdb.set_trace()
    kernel4 = _variable_with_weight_decay('weights3',
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv4 = tf.nn.conv2d(pool3, kernel4, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases4 = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation4 = tf.nn.bias_add(conv4, biases4)
    # conv1 = tf.nn.relu(pre_activation)
    local4 = tf.nn.relu(pre_activation4)
    # pool1
    pool4 = tf.nn.max_pool(local4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')

with tf.name_scope('conv-5') as scope:
#    pdb.set_trace()
    kernel5 = _variable_with_weight_decay('weights3',
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv5 = tf.nn.conv2d(pool4, kernel5, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases5 = tf.Variable(tf.zeros([conv1_out_size]), 'biases') 
    pre_activation5 = tf.nn.bias_add(conv5, biases5)
    # conv1 = tf.nn.relu(pre_activation)
    local5 = tf.nn.relu(pre_activation5)
    # pool1
    pool5 = tf.nn.max_pool(local5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
 
# Move everything into depth so we can perform a single matrix multiply.
with tf.name_scope('Dense-Relu_Layer') as scope:
    # flattening the input
    last_layer = pool5
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
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)
    
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
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            # calculating error and accuracy for batch
            running_error += batch_error
            running_accuracy += batch_acc
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
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            print('err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
            acc_valids.append(valid_accuracy)
            err_valids.append(valid_error)

list_to_file(err_train_list,"res_"+experiment_name+"_error_trains_model9.txt")
list_to_file(acc_train_list,"res_"+experiment_name+"_acc_trains_model9.txt")
list_to_file(err_valids,"res_"+experiment_name+"error_valid_model9.txt")
list_to_file(acc_valids,"res_"+experiment_name+"acc_valid_model9.txt")
