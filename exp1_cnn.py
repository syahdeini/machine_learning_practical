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


def _variable_with_weight_decay(_shape, _mean,_stddev):
    initial = tf.truncated_normal(shape = _shape, mean=_mean, stddev=_stddev)
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
                                         [5, 5, 3, conv1_out_size],
                                         stddev,
                                         mean)

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
                                         [5, 5, conv1_out_size, conv1_out_size],
                                         stddev,
                                         mean)

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
    weights = _variable_with_weight_decay([tot_shape, tot_shape],sttdev,mean)
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


slr = np.linspace(0.0001,0.005,2)
smean = np.linspace(0,0.5,2)
sstd = np.linspace(0.1,1.0,2)

for _lr in slr:
    for _mean in smean:
        for _std in sstd:


            acc_train_list = []
            err_train_list = []
            acc_valids = []
            err_valids = []
            with tf.Session() as sess:
                sess.run(init)
                for e in range(120):
                    running_error = 0.
                    running_accuracy = 0.
                    count=0
                    for input_batch, target_batch in train_data:            
                        # running sesssion
                        #print("shape ",input_batch.shape)
                    #input_batch = normalize_and_whitening(input_batch)
                        #input_batch = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_batch)
                        #count+=1
                    #print("finish normalizeing")
                        _, batch_error, batch_acc = sess.run(
                            [train_step, error, accuracy], 
                          feed_dict={inputs: input_batch, targets: target_batch,mean:_mean,stddev:_std,lrate:_lr})
                        # calculating error and accuracy for batch
                        running_error += batch_error
                        running_accuracy += batch_acc
                    #print("count"+str(count))
                   
                    # averaging the error and accuracy
                    print("calculating error")
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
                                feed_dict={inputs: input_batch, targets: target_batch,mean:_mean,stddev:_std,lrate:_lr})
                            valid_error += batch_error
                            valid_accuracy += batch_acc
                        valid_error /= valid_data.num_batches
                        valid_accuracy /= valid_data.num_batches
                        print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                               .format(valid_error, valid_accuracy))
                        acc_valids.append(valid_accuracy)
                        err_valids.append(valid_error)

            list_to_file(err_train_list,"res_"+"_"+str(_mean)+"_"+str(_std)+"_"+str(_lr)+"_"+experiment_name+"_error_trains_model9.txt")
            list_to_file(acc_train_list,"res_"+"_"+str(_mean)+"_"+str(_std)+"_"+str(_lr)+"_"+experiment_name+"_acc_trains_model9.txt")
            list_to_file(err_valids,"res_"+"_"+str(_mean)+"_"+str(_std)+"_"+str(_lr)+"_"+experiment_name+"error_valid_model9.txt")
            list_to_file(acc_valids,"res_"+"_"+str(_mean)+"_"+str(_std)+"_"+str(_lr)+"_"+experiment_name+"acc_valid_model9.txt")
