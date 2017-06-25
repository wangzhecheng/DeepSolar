"""Author: Zhecheng Wang"""
"""Evaluate Inception-v3 model on benchmark."""
import sys
sys.path.append('/home/ubuntu/vgg_data/code')

import copy
from datetime import datetime
import os.path
import re
import time
import pickle
import csv

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

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/vgg_data/ckpt/inception_final_tuning_3',
                           """Directory where to read model checkpoints.""")

BATCH_SIZE = 100
IMAGE_SIZE = 299
NUM_CLASSES = 2
THRESHOLD = 0.5
EVAL_SET_DIR = '/home/ubuntu/vgg_data/code/generate_dataset/eval_set_list_2.pickle'
RESULT_DIR = '/home/ubuntu/vgg_data/code/inception/eval_result_2'

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

def generate_eval_set():
    # load all train data and return a deque contains all images
    # and corresponding labels.
    with open(EVAL_SET_DIR, 'r') as f:
        eval_set_list = pickle.load(f)
    print('Eval set size: ' + str(len(eval_set_list)))

    eval_set_queue = deque(eval_set_list)

    return eval_set_queue

def test():
    eval_set_queue = generate_eval_set()

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    with tf.Graph().as_default() as g:
        img_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        logits, _ = inception.inference(img_placeholder, NUM_CLASSES)

        saver = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))

        with sess:
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint loaded')
            else:
                print('No checkpoint file found')

            FP_list = []
            FN_list = []
            TP_list = []
            TP_2_list = []
            result_dict = []
            prob_dict = {}
            # initialize
            for ind in xrange(1, 71):
                result_dict.append([ind, 0, 0, 0, 0]) #[TP, TN, FP, FN]
                prob_dict[ind] = []

            for step in xrange(1, 1006):
                start_time = time.time()
                # load data
                minibatch = []
                for count in xrange(0, BATCH_SIZE):
                    element = eval_set_queue.pop()
                    minibatch.append(element)
                    #eval_set_queue.appendleft(element)

                image_list = [load_image(d[0]) for d in minibatch]
                image_path_list = [d[0] for d in minibatch]
                label_list = [d[1] for d in minibatch]
                index_list = [d[2] for d in minibatch]
                step_list = [d[3] for d in minibatch]
                type_list = [d[4] for d in minibatch]

                image_batch = np.array(image_list)

                image_batch = np.reshape(image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

                score = sess.run(logits, feed_dict={img_placeholder: image_batch})

                pos_score = np.exp(score[:, 1])/(np.exp(score[:, 1])+np.exp(score[:, 0]))

                for i in xrange(BATCH_SIZE):
                    prob_dict[index_list[i]].append([label_list[i][0], pos_score[i]])
                    if label_list[i][0] == 1 and pos_score[i] >= THRESHOLD: #TP
                        result_dict[index_list[i]-1][1] += 1
                        if index_list[i] >= 6:
                            TP_list.append([index_list[i], step_list[i], "TP"])
                    elif label_list[i][0] == 1 and pos_score[i] < THRESHOLD: # FN
                        result_dict[index_list[i]-1][4] += 1
                        if index_list[i] >= 6:
                            FN_list.append([index_list[i], step_list[i], "FN"])
                    elif label_list[i][0] == 0 and pos_score[i] < THRESHOLD: # TN
                        result_dict[index_list[i]-1][2] += 1
                    elif label_list[i][0] == 0 and pos_score[i] >= THRESHOLD: # FP
                        result_dict[index_list[i]-1][3] += 1
                        if index_list[i] >= 6:
                            FP_list.append([index_list[i], step_list[i], "FP"])
                    elif label_list[i][0] == 2 and pos_score[i] >= THRESHOLD: #TP
                        result_dict[index_list[i]-1][1] += 1
                        if index_list[i] >= 6:
                            TP_2_list.append(image_path_list[i])
                    elif label_list[i][0] == 2 and pos_score[i] < THRESHOLD: #TN
                        result_dict[index_list[i]-1][2] += 1

                duration = time.time() - start_time

                print("Batch " + str(step) + ", Duration: " + str(duration)+ "s, # images left: " + str(len(eval_set_queue)))

            with open(os.path.join(RESULT_DIR, "prob_list.pickle"), 'w') as f:
                pickle.dump(prob_dict, f)

            with open(os.path.join(RESULT_DIR, "FP_list.pickle"), 'w') as f:
                pickle.dump(FP_list, f)

            with open(os.path.join(RESULT_DIR, "FN_list.pickle"), 'w') as f:
                pickle.dump(FN_list, f)

            with open(os.path.join(RESULT_DIR, "TP_list.pickle"), 'w') as f:
                pickle.dump(TP_list, f)

            with open(os.path.join(RESULT_DIR, "TP_2_list.pickle"), 'w') as f:
                pickle.dump(TP_2_list, f)

            # write csv
            with open(os.path.join(RESULT_DIR, "eval_result.csv"), 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['region', 'TP', 'TN', 'FP', 'FN'])
                writer.writerows(result_dict)
            f.close()


if __name__ == '__main__':
  test()
  print "end"







