# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import pdb
import cPickle


##### CONSTANT #####################################################


experiment_name = "5_layer_conv"
NUM_CLASSES = 10
BATCH_SIZE = 40
train_data = CIFAR10DataProvider('train', batch_size=BATCH_SIZE)
train_data.inputs = train_data.inputs.reshape((-1, 32, 32, 3))
valid_data = CIFAR10DataProvider('valid', batch_size=BATCH_SIZE)
valid_data.inputs = valid_data.inputs.reshape((-1, 32, 32, 3))

# place holder for input and target
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1], train_data.inputs.shape[2], train_data.inputs.shape[3]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')

#################################################################################

def list_to_file(thelist,filename):
#    thefile = open(filename, 'w')
#    for item in thelist:
#      thefile.write("%s\n" % item)
      cPickle.dump(thelist,open(filename,'wb'))


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



def getActivations(layer,stimuli,filename):
    stimuli = stimuli[0].reshape(-1,32,32,3)
    units = sess.run(layer,feed_dict={inputs:stimuli})
    pdb.set_trace()
    units.reshape(1, units.shape[1]*units.shape[2]*units.shape[3])
    list_to_file(units,filename)
    # units.reshape([1,units.shape[1]*units.shape[2]*units.shape[3]])
    # plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        # plt.plot(units[0,:,:,i])
        # plt.savefig("hahaha.png")
        # plt.imshow(units[0,:,:,i],interpolation="nearest", cmap="gray")
        # plt.savefig("plot_"+str(i)+".png") #, interpolation="nearest", cmap="gray")
        # plt.clf()
        plt.hist(units[0,:,:,i])
    plt.savefig("hist_"+".png") #, interpolation="nearest", cmap="gray")



def plot_output_layers(stimuli,layers):
  for key in layers:
    conv,relu,pool = layers[key]
    getActivations(conv,stimuli,key[:-1]+'_conv')
    getActivations(relu,stimuli,key[:-1]+'_relu')
    getActivations(relu,stimuli,key[:-1]+'_pool')

#################### building graph  #################################################

layer_propertieS = {}
conv1_out_size = 14 #number of output channel of first convolutional 
with tf.name_scope('conv-1') as scope:
    kernel = _variable_with_weight_decay('weights'+scope,
                                         shape=[5, 5, 3, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME',name="conv_"+scope)
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = tf.Variable(tf.zeros([conv1_out_size]), name='biases_'+scope) 
    pre_activation = tf.nn.bias_add(conv, biases)
    # conv1 = tf.nn.relu(pre_activation)
    local1 = tf.nn.relu(pre_activation,name="local_relu_"+scope)
    # pool1
    pool1 = tf.nn.max_pool(local1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME',name="maxpool_"+scope)
    layer_propertieS[scope]=(conv,local1,pool1)
with tf.name_scope('conv-2') as scope:
    kernel2 = _variable_with_weight_decay('weights2'+scope,
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME',name="conv_"+scope)
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([conv1_out_size]), name='biases_'+scope) 
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
    local2 = tf.nn.relu(pre_activation2, name='local_relu_'+scope)
    # pool1
    pool2 = tf.nn.max_pool(local2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME',name="maxpool_"+scope)

    layer_propertieS[scope]=(conv2,local2,pool2)
with tf.name_scope('conv-3') as scope:
    kernel3 = _variable_with_weight_decay('weights3'+scope,
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME',name="conv_"+scope)
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases3 = tf.Variable(tf.zeros([conv1_out_size]), name='biases_'+scope) 
    pre_activation3 = tf.nn.bias_add(conv3, biases3)
    # conv1 = tf.nn.relu(pre_activation)
    local3 = tf.nn.relu(pre_activation3, name='local_relu_'+scope)
    # pool1
    pool3 = tf.nn.max_pool(local3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME',name="maxpool_"+scope)
    layer_propertieS[scope]=(conv3,local3,pool3)

with tf.name_scope('conv-4') as scope:
    kernel4 = _variable_with_weight_decay('weights4'+scope,
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv4 = tf.nn.conv2d(pool3, kernel4, [1, 1, 1, 1], padding='SAME',name="conv_"+scope)
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases4 = tf.Variable(tf.zeros([conv1_out_size]), name='biases_'+scope) 
    pre_activation4 = tf.nn.bias_add(conv4, biases4)
    # conv1 = tf.nn.relu(pre_activation)
    local4 = tf.nn.relu(pre_activation4, name='local_relu_'+scope)
    # pool1
    pool4 = tf.nn.max_pool(local4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME',name="maxpool_"+scope)
    layer_propertieS[scope]=(conv4,local4,pool4)

with tf.name_scope('conv-5') as scope:
    kernel5 = _variable_with_weight_decay('weights5'+scope,
                                         shape=[5, 5, conv1_out_size, conv1_out_size],
                                         stddev=5e-2,
                                         wd=0.0)

    conv5 = tf.nn.conv2d(pool4, kernel5, [1, 1, 1, 1], padding='SAME',name="conv_"+scope)
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases5 = tf.Variable(tf.zeros([conv1_out_size]), 'biases', name='biases_'+scope) 
    pre_activation5 = tf.nn.bias_add(conv5, biases5)
    # conv1 = tf.nn.relu(pre_activation)
    local5 = tf.nn.relu(pre_activation5, name='local_relu_'+scope)
    # pool1
    pool5 = tf.nn.max_pool(local5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME',name="maxpool_"+scope)
    
    layer_propertieS[scope]=(conv5,local5,pool5)

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
            tf.equal(tf.argmax(soft_max_out, 1), tf.argmax(targets, 1)), 
            tf.float32))


# use adam optimizer 
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)
    
init = tf.global_variables_initializer()
# begin training

#########################################################################

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
        if  (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                #input_batch=tf.reshape(input_batch,[BATCH_SIZE,32,32,3])
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
                plot_output_layers(input_batch,layer_propertieS)

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
