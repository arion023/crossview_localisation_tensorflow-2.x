#This is cut-out from train.py file to only evaluate models

from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputData

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = 'CVM-NET-I'

# path to model.ckpt file
load_model_path = ''

is_training = False

#TODO it can be elimated
batch_size = 12

# -------------------------------------------------------- #


def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy

# import data
input_data = InputData()

# define placeholders
sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
keep_prob = tf.placeholder(tf.float32)

# build model
if network_type == 'CVM-NET-I':
    sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
elif network_type == 'CVM-NET-II':
    sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)
else:
    print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')

saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

# run model
print('run model...')
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print('load model...')
    saver.restore(sess, load_model_path)
    print("   Model loaded from: %s" % load_model_path)
    print('load model...FINISHED')

# ---------------------- validation ----------------------
    print('validate...')
    print('   compute global descriptors')
    input_data.reset_scan()
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
    val_i = 0
    while True:
        print('      progress %d' % val_i)
        batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
        if batch_sat is None:
            break
        feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
        sat_global_val, grd_global_val = \
            sess.run([sat_global, grd_global], feed_dict=feed_dict)

        sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
        grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
        val_i += sat_global_val.shape[0]

    print('   compute accuracy')
    val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
    print('model :  ' + load_model_path)
    print('\taccuracy = %.1f%%' % ( val_accuracy*100.0))

    # ---------------------------------------------------------