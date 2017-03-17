# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
from maxout import max_out
import pdb


experiment_name = "nin_"

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

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


num_hidden = 200
num_hidden2 = 100
num_hidden3 = 50
BATCH_SIZE = 50
NUM_CLASSES = 10
train_data = CIFAR10DataProvider('train', batch_size=BATCH_SIZE)
train_data.inputs = train_data.inputs.reshape((-1, 32, 32, 3))
valid_data = CIFAR10DataProvider('valid', batch_size=BATCH_SIZE)
valid_data.inputs = valid_data.inputs.reshape((-1, 32, 32, 3))
input_dim = 32
output_dim = 32
# place holder for input and target
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1], train_data.inputs.shape[2], train_data.inputs.shape[3]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
# pdb.set_trace()
# building graph

conv1_out_size = 14 #number of output channel of first convolutional 
#CONV1
with tf.name_scope('conv-1') as scope:
    kernel1 = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 192],
                                         stddev=5e-2,
                                         wd=0.0)

    conv1 = tf.nn.conv2d(inputs, kernel1, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases1 = tf.Variable(tf.zeros([192]), 'biases') 
    pre_activation1 = tf.nn.bias_add(conv1, biases1)
    # conv1 = tf.nn.relu(pre_activation)
    local1 = tf.nn.relu(pre_activation1)
    # pool1

with tf.name_scope('cccp-1') as scope:
    # kernel_cp1 = _variable_with_weight_decay('weights_cccp1',
    #                                      shape=[1, 1, 192, 160],
    #                                      stddev=5e-2,
    #                                      wd=1.0)

    # conv_cp1 = tf.nn.conv2d(local1, kernel_cp1, [1, 1, 1, 1], padding='SAME')
    # #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    # biases_cp1 = tf.Variable(tf.zeros([160]), 'biases') 
    # pre_activation_cp1 = tf.nn.bias_add(conv_cp1, biases_cp1)
    # # conv1 = tf.nn.relu(pre_activation)
    # # local_cp1 = tf.nn.relu(pre_activation_cp1)
    bn1_mean_cp1, bn1_var_cp1 = tf.nn.moments(local1,[0])
    bn1_cp1 = tf.nn.batch_normalization(local1, bn1_mean_cp1, bn1_var_cp1)

    shape_pool_cp1 = bn1_cp1.get_shape().as_list()
    out_dim_cp1 = shape_pool_cp1[1]*shape_pool_cp1[2]*shape_pool_cp1[3]
    # pdb.set_trace()
    reshape_cp1 = tf.reshape(bn1_cp1, [BATCH_SIZE,out_dim_cp1])
   # dim = reshape.get_shape()[1].value
    weights_cp1 = _variable_with_weight_decay('weights_ccp1', shape=[out_dim_cp1,160],
                                          stddev=0.05, wd=0.0)
    biases_cp1 = tf.Variable(tf.zeros([160]), 'biases')   
    local_cp1 = max_out(tf.matmul(reshape_cp1, weights_cp1) + biases_cp1).reshape(shape_pool_cp1)
    
    # BN and Maxout 2
    bn2_mean_cp1, bn2_var_cp1 = tf.nn.moments(local_cp1,[0])
    bn2_cp1 = tf.nn.batch_normalization(local_cp1, bn2_mean_cp1, bn2_var_cp1)


    shape2_pool_cp1 = bn1_cp1.get_shape().as_list()
    out_dim2_cp1 = shape2_pool_cp1[1]*shape2_pool_cp1[2]*shape2_pool_cp1[3]
    # pdb.set_trace()
    reshape2_cp1 = tf.reshape(bn1_cp1, [BATCH_SIZE,out_dim2_cp1])
    # dim = reshape.get_shape()[1].value
    weights2_cp1 = _variable_with_weight_decay('weights_ccp1', shape=[out_dim2_cp1,160],
                                          stddev=0.05, wd=0.0)
    biases2_cp1 = tf.Variable(tf.zeros([160]), 'biases')   
    local2_cp1 = max_out(tf.matmul(reshape2_cp1, weights2_cp1) + biases2_cp1).reshape(shape2_pool_cp1)


    # BN and Maxout 3
    bn3_mean_cp1, bn3_var_cp1 = tf.nn.moments(local2_cp1,[0])
    bn3_cp1 = tf.nn.batch_normalization(local2_cp1, bn3_mean_cp1, bn3_var_cp1)


    shape3_pool_cp1 = bn3_cp1.get_shape().as_list()
    out_dim3_cp1 = shape3_pool_cp1[1]*shape3_pool_cp1[2]*shape3_pool_cp1[3]
    # pdb.set_trace()
    reshape3_cp1 = tf.reshape(bn3_cp1, [BATCH_SIZE,out_dim3_cp1])
    # dim = reshape.get_shape()[1].value
    weights3_cp1 = _variable_with_weight_decay('weights_ccp1', shape=[out_dim3_cp1,160],
                                          stddev=0.05, wd=0.0)
    biases3_cp1 = tf.Variable(tf.zeros([160]), 'biases')   
    local3_cp1 = max_out(tf.matmul(reshape3_cp1, weights3_cp1) + biases3_cp1).reshape(shape3_pool_cp1)



    # pool1
with tf.name_scope('max_pool_conv1') as scope:
    pool1 = tf.nn.max_pool(local_cp2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
with tf.name_scope('dropout_conv1') as scope:
    dropout1 = tf.nn.dropout(pool1,keep_prob=0.5,name="dropout1") 

#CONV2
with tf.name_scope('conv-2') as scope:
    kernel2 = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 96, 192],
                                         stddev=5e-2,
                                         wd=0.0)

    conv2 = tf.nn.conv2d(dropout1, kernel2, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([192]), 'biases') 
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
    local2 = tf.nn.relu(pre_activation2)
    # pool1

# # with tf.name_scope('cccp-2') as scope:
# #     pass





# # with tf.name_scope('avg_pool_conv2') as scope:
# #     pool2 = tf.nn.avg_pool(local_cp4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
# #                          padding='SAME')
# # with tf.name_scope('dropout_conv2') as scope:
# #     dropout2 = tf.nn.dropout(pool2,keep_prob=0.5,name="dropout2") 

# # with tf.name_scope('conv-3') as scope:
# #     pdb.set_trace()
# #     kernel3 = _variable_with_weight_decay('weights',
# #                                          shape=[3, 3, 96/2, 192],
# #                                          stddev=5e-2,
# #                                          wd=0.0)

# #     conv3 = tf.nn.conv2d(dropout2, kernel3, [1, 1, 1, 1], padding='SAME')
# #     #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
# #     biases3 = tf.Variable(tf.zeros([192]), 'biases') 
# #     pre_activation3 = tf.nn.bias_add(conv3, biases3)
# #     # conv1 = tf.nn.relu(pre_activation)
# #     local3 = tf.nn.relu(pre_activation3)
# #     # pool1

# # with tf.name_scope('cccp-5') as scope:
# #     kernel_cp6 = _variable_with_weight_decay('weights_cccp5',
# #                                          shape=[1, 1, 192, 192],
# #                                          stddev=5e-2,
# #                                          wd=0.0)

# #     conv_cp6 = tf.nn.conv2d(local3, kernel_cp6, [1, 1, 1, 1], padding='SAME')
# #     #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
# #     biases_cp6 = tf.Variable(tf.zeros([10]), 'biases') 
# #     pre_activation_cp6 = tf.nn.bias_add(conv_cp6, biases_cp6)
# #     # conv1 = tf.nn.relu(pre_activation)
# #     local_cp6 = tf.nn.relu(pre_activation_cp6)
# # with tf.name_scope('cccp-6') as scope:
# #     kernel_cp6 = _variable_with_weight_decay('weights_cccp6',
# #                                          shape=[1, 1, 192, 10],
# #                                          stddev=5e-2,
# #                                          wd=0.0)

# #     conv_cp6 = tf.nn.conv2d(local_cp6, kernel_cp6, [1, 1, 1, 1], padding='SAME')
# #     #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
# #     biases_cp6 = tf.Variable(tf.zeros([10]), 'biases') 
# #     pre_activation_cp6 = tf.nn.bias_add(conv_cp6, biases_cp6)
# #     # conv1 = tf.nn.relu(pre_activation)
# #     local_cp6 = tf.nn.relu(pre_activation_cp6)
# #     # pool1
# with tf.name_scope('avg_pool_conv3') as scope:
#     pool3 = tf.nn.max_pool(local_cp6, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1],
#                          padding='SAME')



with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights5', [10, NUM_CLASSES],
                                          stddev=1.0, wd=0.0)
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), 'biases') 
    softmax_linear = tf.add(tf.matmul(pool3, weights), biases)
    soft_max_out = tf.nn.softmax(softmax_linear)

with tf.name_scope('error'):
    out_login = tf.nn.softmax_cross_entropy_with_logits(logits = softmax_linear, labels=targets)
    error = tf.reduce_mean(out_login)

# use softmax for accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(soft_max_out, 1), tf.argmax(targets, 1)), 
            tf.float32))

# use adam optimizer 
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(error)
    
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
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
            acc_valids.append(valid_accuracy)
            err_valids.append(valid_error)

list_to_file(err_train_list,"res_"+experiment_name+"_error_trains_model9.txt")
list_to_file(acc_train_list,"res_"+experiment_name+"_acc_trains_model9.txt")
list_to_file(err_valids,"res_"+experiment_name+"error_valid_model9.txt")
list_to_file(acc_valids,"res_"+experiment_name+"acc_valid_model9.txt")
