"""Author: Zhecheng Wang"""
"""Classfication and localization of Inception-v3 model on benchmark."""
import sys
sys.path.append('/home/ubuntu/vgg_data/code')

import copy
from datetime import datetime
import os.path
import re
import time
import pickle
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import random
import pickle
import math
from collections import deque

# from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/vgg_data/test_set/benchmark2',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/vgg_data/ckpt/inception_2',
                           """Directory where to read model checkpoints.""")

HEAT_MAP_DIR_1 = '/home/ubuntu/vgg_data/code/inception/eval_seg_results/TP/'
CKPT_DIR_1 = '/home/ubuntu/vgg_data/ckpt/inception_final_tuning_3'
CKPT_DIR_2 = '/home/ubuntu/vgg_data/ckpt/detector_2_layers'
SEG_DATA_DIR = '/home/ubuntu/vgg_data/segmentation_data/seg_eval_set/TP'
#weights = np.load('/home/ubuntu/vgg_data/code/inception/weight_value_1000.npy')
# basic parameters
BATCH_SIZE = 32
IMAGE_SIZE = 299
NUM_CLASSES = 2
# test sample range
RANGE = 3000
# region_list
REGION_LIST = ['sunset', 'neal', 'beresford_park', 'south_of_market', 'santa_clara', 'sunnyvale']
#REGION_LIST = ['sunset', 'south_of_market', 'presidio', 'beresford_park', 'neal', 'santa_clara', 'sunnyvale', 'houston', 'boston', 'charlotte']
def load_image(path):
    # load image and prepocess, solar_panel is a bool: True if it is solar panel.
    # rotate_angle_list = [0, 90, 180, 270]
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    # rotate_angle = random.choice(rotate_angle_list)
    # image = skimage.transform.rotate(resized_img, rotate_angle)
    return resized_img

def save_heat_map(step, classmap_val):
    # generate class activation map and save.
    np.save(HEAT_MAP_DIR_1 + str(step) + '_CAM.npy', classmap_val[0])
    classmap_vis = map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_val)
    vis = classmap_vis[0]
    skimage.io.imsave(HEAT_MAP_DIR_1 + str(step) + '_CAM.png', vis)

def test():
    #os.mkdir(HEAT_MAP_DIR_1)
    with tf.Graph().as_default() as g:
        img_placeholder = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3])

        logits, _, feature_map = inception.inference(img_placeholder, NUM_CLASSES)

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

        aux_logits = tf.matmul(GAP, W2)

        # classmap = inception.get_classmap(1, conv_map)
        conv_map_resized = tf.image.resize_bilinear(conv_aux_2, [100, 100])

        label_w = tf.gather(tf.transpose(W2), 1)
        label_w = tf.reshape(label_w, [-1, 512, 1])
        conv_map_resized = tf.reshape(conv_map_resized, [-1, 100 * 100, 512])
        classmap = tf.batch_matmul(conv_map_resized, label_w)
        classmap = tf.reshape(classmap, [-1, 100, 100])

        # Construct saver
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        print variables_to_restore
        saver1 = tf.train.Saver(variables_to_restore)
        saver2 = tf.train.Saver(var_list=[W2, kernel2, biases2, kernel1, biases1])

        with tf.Session() as sess:

            checkpoint1 = tf.train.get_checkpoint_state(CKPT_DIR_1)
            if checkpoint1 and checkpoint1.model_checkpoint_path:
                saver1.restore(sess, checkpoint1.model_checkpoint_path)
                print("Successfully loaded:", checkpoint1.model_checkpoint_path)
            else:
                print("Could not find old network weights")

            checkpoint2 = tf.train.get_checkpoint_state(CKPT_DIR_2)
            if checkpoint2 and checkpoint2.model_checkpoint_path:
                saver2.restore(sess, checkpoint2.model_checkpoint_path)
                print("Successfully loaded:", checkpoint2.model_checkpoint_path)
            else:
                print("Could not find old network weights")

            TP = FN = 0
            for step in xrange(1, 1100):
                image_path = os.path.join(SEG_DATA_DIR, '%d.png' % step)
                if not os.path.exists(image_path):
                    continue
                image = load_image(image_path)
                img_batch = np.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
                score = sess.run(logits, feed_dict={img_placeholder: img_batch})
                prob = math.exp(score[0, 1]) / (math.exp(score[0, 1]) + math.exp(score[0, 0]))
                if prob >= 0.5:
                    TP += 1
                    classmap_val = sess.run(classmap, feed_dict={img_placeholder: img_batch})
                    save_heat_map(step, classmap_val)
                else:
                    FN += 1

                print step
            print ("TP: "+str(TP)+", FN: "+str(FN))


if __name__ == '__main__':
  test()
