# elu tanh smooth
import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import pdb


experiment_name = "imagenet_leaky_without_imagepre"

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
keep_prob = tf.placeholder("float")
reguralization_val = 1.5
conv1_out_size = 14 #number of output channel of first convolutional
def resize_img(imgs):
    return tf.image.resize_images(inputs, [16,16])

def distorted_image(image):
   bounding_boxes = [BATCH_SIZE,3,4]

   # Generate a single distorted bounding box.
   begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
                       tf.shape(image),
                       bounding_boxes=bounding_boxes)

   # Draw the bounding box in an image summary.
   image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                 bbox_for_draw)
   tf.image_summary('images_with_box', image_with_box)
   # Employ the bounding box to distort the image.
   distorted_image = tf.slice(image, begin, size)

def random_aug(image):
    image = tf.image.rgb_to_grayscale(image)
    distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
    distort_left_right_random = distortions[0]
    mirror = tf.less(tf.pack([1.0, distort_left_right_random, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    distort_up_down_random = distortions[1]
    mirror = tf.less(tf.pack([distort_up_down_random, 1.0, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    return image

alpha = float(1)/100   
with tf.name_scope('conv-1') as scope:
    filter_size1 = 48
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, filter_size1],
                                         stddev=5e-2,
                                         wd=0.5)
  #  inp = resize_img(inputs)
     #tfn = lambda x: random_aug(x)
    # inp = tf.map_fn(fn=tfn, elems = inp)
    tfn = lambda x: tf.image.per_image_standardization(x)
    inp = tf.map_fn(fn=random_aug, elems=inputs)
   # pdb.set_trace()
    conv1 = tf.nn.conv2d(inp, kernel, [1, 1, 1, 1], padding='SAME')    
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = tf.Variable(tf.zeros([filter_size1]), 'biases') 
    pre_activation = tf.nn.bias_add(conv1, biases)
    # conv1 = tf.nn.relu(pre_activation)
    #local1 = tf.nn.relu(pre_activation)
    local1 = tf.maximum(alpha*pre_activation,pre_activation)
    
    # pool1
    lrn_out = tf.nn.lrn(local1, 5,2,0.0001,0.75)
with tf.name_scope('conv-2') as scope:
#    pdb.set_trace()
    filter_size2 = 128
    kernel2 = _variable_with_weight_decay('weights2',
                                         shape=[5, 5, filter_size1, filter_size2],
                                         stddev=5e-2,
                                         wd=0.5)

    conv2 = tf.nn.conv2d(lrn_out, kernel2, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases2 = tf.Variable(tf.zeros([filter_size2]), 'biases') 
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    # conv1 = tf.nn.relu(pre_activation)
   # local2 = tf.nn.relu(pre_activation2)
    # pool1
    local2 = tf.maximum(alpha*pre_activation2,pre_activation2)

    pool2 = tf.nn.max_pool(local2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')
with tf.name_scope('dropout') as scope:
    local_dp1 = tf.nn.dropout(pool2, keep_prob)

with tf.name_scope('conv-3') as scope:
#    pdb.set_trace()
    filter_size3 = 192
    kernel3 = _variable_with_weight_decay('weights3',
                                         shape=[5, 5, filter_size2, filter_size3],
                                         stddev=5e-2,
                                         wd=0.5)

    conv3 = tf.nn.conv2d(local_dp1, kernel3, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases3 = tf.Variable(tf.zeros([filter_size3]), 'biases') 
    pre_activation3 = tf.nn.bias_add(conv3, biases3)
    # conv1 = tf.nn.relu(pre_activation)
    #local3 = tf.nn.relu(pre_activation3)
    # pool1

    local3 = tf.maximum(alpha*pre_activation3,pre_activation3)

with tf.name_scope('conv-4') as scope:
#    pdb.set_trace()
    filter_size4 = 192
    kernel4 = _variable_with_weight_decay('weights4',
                                         shape=[5, 5, filter_size3, filter_size4],
                                         stddev=5e-2,
                                         wd=0.5)

    conv4 = tf.nn.conv2d(local3, kernel4, [1, 1, 1, 1], padding='SAME')
    #biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases4 = tf.Variable(tf.zeros([filter_size4]), 'biases')
    pre_activation4 = tf.nn.bias_add(conv4, biases4)
    local4 = tf.maximum(alpha*pre_activation4,pre_activation4)

    pool4 = tf.nn.max_pool(local4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME')

with tf.name_scope('dropou2t') as scope:
    local_dp2 = tf.nn.dropout(pool4, keep_prob)

 
# Move everything into depth so we can perform a single matrix multiply.
with tf.name_scope('Dense-Relu_Layer') as scope:
    # flattening the input
    last_layer = local_dp2
    tot_shape=last_layer.get_shape()[1].value*last_layer.get_shape()[2].value*last_layer.get_shape()[3].value
    reshape = tf.reshape(last_layer, [BATCH_SIZE,tot_shape])
    weights = _variable_with_weight_decay('weights3', shape=[tot_shape, tot_shape],stddev=1.0, wd=0.5)
    biases = tf.Variable(tf.zeros([tot_shape]), 'biases') 
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

with tf.name_scope('Dense-Relu_Layer_2') as scope:
    # flattening the input
    last_layer = local3
#    reshape_dl2 = tf.reshape(last_layer, [BATCH_SIZE,tot_shape])
    weights_dl2 = _variable_with_weight_decay('weights3', shape=[tot_shape, tot_shape],stddev=1.0, wd=0.5)
    biases_dl2 = tf.Variable(tf.zeros([tot_shape]), 'biases') 
    local_dl2 = tf.nn.relu(tf.matmul(last_layer, weights_dl2) + biases_dl2)


with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights5', [tot_shape, NUM_CLASSES],
                                          stddev=1.0, wd=0.0)
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), 'biases') 
    softmax_linear = tf.add(tf.matmul(local_dl2, weights), biases)
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
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob:0.5})
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
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob:1.0})
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
