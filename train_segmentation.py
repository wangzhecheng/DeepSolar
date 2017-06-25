"""
Greedy layerwise training for weakly-supervised solar panel segmentation.
We need to train first layer in the branch and then train the second layer
but fix the first layer. FLAGS.two_layers determines which mode to run.
"""

import sys
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

from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('two_layers', False,
                           """ If true, two layers in the segmentation branch,
                           else one layer. """)

tf.app.flags.DEFINE_string('ckpt_save_dir', 'ckpt/inception_segmentation',
                           """Directory for saving model checkpoint. """)

tf.app.flags.DEFINE_string('ckpt_restore_dir', 'ckpt/inception_segmentation',
                           """Directory for restoring first layer. """)

tf.app.flags.DEFINE_string('classification_ckpt_restore_dir', 'ckpt/inception_classification',
                           """ Directory for restoring parameters of classification model. """)

tf.app.flags.DEFINE_integer('max_steps', 15000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_float('learning_rate', 0.005,
                          """learning rate.""")


# basic parameters
BATCH_SIZE = 64
IMAGE_SIZE = 299
NUM_CLASSES = 2

def load_image(path):
    # load and prepocess image
    rotate_angle_list = [0, 90, 180, 270]
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    rotate_angle = random.choice(rotate_angle_list)
    image = skimage.transform.rotate(resized_img, rotate_angle)
    return image

def train():
    # load train set list and transform it to queue. For time concern, we recommmend use a subset of training set
    # to fine-tune the segmentation branch.
    try:
        with open('train_set_list.pickle', 'r') as f:
            train_set_list = pickle.load(f)
    except:
        raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
    random.shuffle(train_set_list)
    train_set_queue = deque(train_set_list)
    train_set_size = len(train_set_list)
    del train_set_list
    print ('Training set built. Size: '+str(train_set_size))

    # build the tensorflow graph
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        # get the feature map after layer 'mixed_35x35x288b' from classification model.
        _, _, feature_map = inception.inference(images, NUM_CLASSES)

        with tf.name_scope('conv_aux_1') as scope:
            kernel1 = tf.Variable(tf.truncated_normal([3, 3, 288, 512], dtype=tf.float32, stddev=1e-4), name='weights')
            conv = tf.nn.conv2d(feature_map, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases1)
            conv_aux = tf.nn.relu(bias, name=scope)

        if FLAGS.two_layers:
            with tf.name_scope('conv_aux_2') as scope:
                kernel2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-4), name='weights')
                conv = tf.nn.conv2d(conv_aux, kernel2, [1, 1, 1, 1], padding='SAME')
                biases2 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases2)
                conv_aux = tf.nn.relu(bias, name=scope)

        # global average pooling.
        GAP = tf.reduce_mean(conv_aux, [1, 2])

        # linear classifier.
        W = tf.get_variable(name='W', shape=[512, 2], initializer=tf.random_normal_initializer(0., 0.01))
        logits = tf.matmul(GAP, W)

        # compute loss.
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

        if FLAGS.two_layers:
            train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,
                                                                                         var_list=[W, kernel2, biases2])
            saver = tf.train.Saver(var_list=[W, kernel2, biases2, kernel1, biases1])
        else:
            train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,
                                                                                         var_list=[W, kernel1, biases1])
            saver = tf.train.Saver(var_list=[W, kernel1, biases1])

        init = tf.global_variables_initializer()

        # open session and initialize
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))
        sess.run(init)

        # restore old checkpoint
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        restorer1 = tf.train.Saver(variables_to_restore)

        checkpoint1 = tf.train.get_checkpoint_state(FLAGS.classification_ckpt_restore_dir)
        if checkpoint1 and checkpoint1.model_checkpoint_path:
            restorer1.restore(sess, checkpoint1.model_checkpoint_path)
            print("Successfully loaded:", checkpoint1.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # If adding second layer, then the first layer should be restored.
        if FLAGS.two_layers:
            restorer2 = tf.train.Saver(var_list=[kernel1, biases1])
            checkpoint2 = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir)
            if checkpoint2 and checkpoint2.model_checkpoint_path:
                restorer2.restore(sess, checkpoint2.model_checkpoint_path)
                print("Successfully loaded:", checkpoint2.model_checkpoint_path)
            else:
                print("Could not find old network weights")

        step = 1
        train_record = []
        while step <= FLAGS.max_steps:
            start_time = time.time()
            # construct image batch and label batch for one step train.
            minibatch = []
            for count in xrange(0, BATCH_SIZE):
                element = train_set_queue.pop()
                minibatch.append(element)
                train_set_queue.appendleft(element)

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
                checkpoint_path = os.path.join(FLAGS.ckpt_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1

if __name__ == '__main__':
    train()
