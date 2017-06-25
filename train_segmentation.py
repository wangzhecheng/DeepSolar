"""Author: Zhecheng Wang"""
"""Fine-tuning the Inception-v3 model for solar panel identification with multi-gpus."""
import sys
sys.path.append('/home/ubuntu/vgg_data/code')

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import random
import pickle
from collections import deque

# from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/vgg_data/ckpt/detector_2_layers_2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/home/ubuntu/vgg_data/train_set',
                           """Directory of training set""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/home/ubuntu/vgg_data/ckpt/inception_2',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.0003,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 7.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")

CKPT_DIR_1 = '/home/ubuntu/vgg_data/ckpt/inception_final_tuning_3'
CKPT_DIR_2 = '/home/ubuntu/vgg_data/ckpt/detector_1_layer'

# basic parameters
BATCH_SIZE = 64
IMAGE_SIZE = 299
NUM_CLASSES = 2
l1_coef = 0.01
# training sample range
RANGE = 30000
# number of training samples
TRAIN_SET_SIZE = 11000
# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 0.1              # Epsilon term for RMSProp.

def load_image(path):
    # load image and prepocess, solar_panel is a bool: True if it is solar panel.
    rotate_angle_list = [0, 90, 180, 270]
    img = skimage.io.imread(path)
    # resize to 100*100
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    rotate_angle = random.choice(rotate_angle_list)
    image = skimage.transform.rotate(resized_img, rotate_angle)
    return image

def generate_train_set():
    # load all train data and return a deque contains all images
    # and corresponding labels.
    image_path = []
    # load negative data
    data_dir_1 = os.path.join(FLAGS.data_dir, '0_1')
    neg_num = 0
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_1, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1
    """data_dir_2 = os.path.join(FLAGS.data_dir, 'old/0_2')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_2, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1"""
    data_dir_3 = os.path.join(FLAGS.data_dir, 'old/0_4')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_3, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1
    """data_dir_4 = os.path.join(FLAGS.data_dir, 'old/0_5')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_4, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1"""
    print(neg_num, " non-panel training images")

    # load positive data
    data_dir_5 = os.path.join(FLAGS.data_dir, '1')
    pos_num = 0
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_5, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [1]))
        pos_num += 1
    data_dir_6 = os.path.join(FLAGS.data_dir, 'old/1')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_6, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [1]))
        pos_num += 1
    print(pos_num, " panel training images")
    # shuffle the collection
    random.shuffle(image_path)
    # build training set and validation set
    train_set = image_path[0: TRAIN_SET_SIZE]
    train_set = deque(train_set)
    print("built a training set with size of ", train_set.__len__())
    val_set = image_path[TRAIN_SET_SIZE: image_path.__len__()]
    val_set = deque(val_set)
    print("built a validation set with size of ", val_set.__len__())
    return train_set, val_set

def train():
    train_set, val_set = generate_train_set()
    with tf.Graph().as_default() as g:
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.

        images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        output, _, feature_map = inception.inference(images, NUM_CLASSES)

        with tf.name_scope('conv_aux_1') as scope:
            kernel1 = tf.Variable(tf.truncated_normal([3, 3, 288, 512], dtype=tf.float32, stddev=1e-4), name='weights')
            conv = tf.nn.conv2d(feature_map, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases1)
            conv_aux_1 = tf.nn.relu(bias, name=scope)

        with tf.name_scope('conv_aux_2') as scope:
            kernel2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-4), name='weights')
            conv = tf.nn.conv2d(conv_aux_1, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases2)
            conv_aux_2 = tf.nn.relu(bias, name=scope)


        GAP = tf.reduce_mean(conv_aux_2, [1, 2])

        W2 = tf.get_variable(name='W', shape=[512, 2], initializer=tf.random_normal_initializer(0., 0.01))

        logits = tf.matmul(GAP, W2)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

        # add l1 regularization to create sparse model
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_coef, scope=None)
        l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, [W2])
        loss = tf.add(cross_entropy, l1_loss)

        train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss, var_list=[W2, kernel2, biases2])

        saver = tf.train.Saver(var_list=[W2, kernel2, biases2, kernel1, biases1])

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # open session and initialize
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))
        sess.run(init)

        # restore old checkpoint
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        restorer1 = tf.train.Saver(variables_to_restore)
        restorer2 = tf.train.Saver(var_list=[kernel1, biases1])

        checkpoint1 = tf.train.get_checkpoint_state(CKPT_DIR_1)
        if checkpoint1 and checkpoint1.model_checkpoint_path:
            restorer1.restore(sess, checkpoint1.model_checkpoint_path)
            print("Successfully loaded:", checkpoint1.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        checkpoint2 = tf.train.get_checkpoint_state(CKPT_DIR_2)
        if checkpoint2 and checkpoint2.model_checkpoint_path:
            restorer2.restore(sess, checkpoint2.model_checkpoint_path)
            print("Successfully loaded:", checkpoint2.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        step = 1
        train_record = []
        while step <= FLAGS.max_steps:
            start_time = time.time()
            # construct image batch and label batch for one step train
            minibatch = random.sample(train_set, BATCH_SIZE)
            image_list = [load_image(d[0]) for d in minibatch]
            label_list = [d[1] for d in minibatch]

            image_batch = np.array(image_list)
            label_batch = np.array(label_list)

            image_batch = np.reshape(image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
            label_batch = np.reshape(label_batch, [BATCH_SIZE])

            _, loss_value = sess.run([train_step, cross_entropy], feed_dict={images: image_batch, labels: label_batch})

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step == 1 or step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')

                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step == 1 or step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # Give training error and validation error every 400 steps
            if step % 1000 == 0:
                print("begin calculating training error at step ", step)
                TP1 = TN1 = FP1 = FN1 = 0
                for ii in xrange(0, 20):
                    minibatch2 = random.sample(train_set, BATCH_SIZE)
                    image_list2 = [load_image(d[0]) for d in minibatch2]
                    label_list2 = [d[1] for d in minibatch2]
                    image_batch2 = np.array(image_list2)
                    label_batch2 = np.array(label_list2)
                    image_batch2 = np.reshape(image_batch2, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
                    label_batch2 = np.reshape(label_batch2, [BATCH_SIZE])
                    score = sess.run(logits, feed_dict={images: image_batch2})
                    for k in xrange(0, BATCH_SIZE):
                        if label_batch2[k] == 1 and score[k, 1] >= score[k, 0]:
                            TP1 += 1
                        elif label_batch2[k] == 1 and score[k, 1] < score[k, 0]:
                            FN1 += 1
                        elif label_batch2[k] == 0 and score[k, 1] <= score[k, 0]:
                            TN1 += 1
                        elif label_batch2[k] == 0 and score[k, 1] > score[k, 0]:
                            FP1 += 1
                precision = float(TP1) / float(TP1 + FP1+0.000001)
                recall = float(TP1) / float(TP1 + FN1+0.000001)
                print("Training set: TP:", TP1, "TN:", TN1, "FP:", FP1, "FN:", FN1, "precision:", precision, "recall:",
                      recall)
                train_record.append(('train', step, TP1, TN1, FP1, FN1, precision, recall))

                print("begin calculating validation error at step ", step)
                TP2 = TN2 = FP2 = FN2 = 0
                for ii in xrange(0, 20):
                    minibatch3 = random.sample(val_set, BATCH_SIZE)
                    image_list3 = [load_image(d[0]) for d in minibatch3]
                    label_list3 = [d[1] for d in minibatch3]
                    image_batch3 = np.array(image_list3)
                    label_batch3 = np.array(label_list3)
                    image_batch3 = np.reshape(image_batch3, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
                    label_batch3 = np.reshape(label_batch3, [BATCH_SIZE])
                    score = sess.run(logits, feed_dict={images: image_batch3})
                    for k in xrange(0, BATCH_SIZE):
                        if label_batch3[k] == 1 and score[k, 1] >= score[k, 0]:
                            TP2 += 1
                        elif label_batch3[k] == 1 and score[k, 1] < score[k, 0]:
                            FN2 += 1
                        elif label_batch3[k] == 0 and score[k, 1] <= score[k, 0]:
                            TN2 += 1
                        elif label_batch3[k] == 0 and score[k, 1] > score[k, 0]:
                            FP2 += 1
                precision2 = float(TP2) / float(TP2 + FP2+0.000001)
                recall2 = float(TP2) / float(TP2 + FN2+0.000001)
                print(
                "Validation set: TP:", TP2, "TN:", TN2, "FP:", FP2, "FN:", FN2, "precision:", precision2, "recall:",
                recall2)
                train_record.append(('val', step, TP2, TN2, FP2, FN2, precision2, recall2))

                with open("record_detector_2_layers_2.pickle", 'w') as f:
                    pickle.dump(train_record, f)

            step += 1



if __name__ == '__main__':
    train()
