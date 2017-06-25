"""Evaluate Inception-v3 model on test(eval) set."""

import sys
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
import pickle
from collections import deque

from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', 'ckpt/inception_classification',
                           """Directory for restoring trained model checkpoints.""")

BATCH_SIZE = 100
IMAGE_SIZE = 299
NUM_CLASSES = 2
THRESHOLD = 0.5 # softmax score threshold of classifying a sample to be positive.

def load_image(path):
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    return resized_img

def generate_eval_set():
    # load all train data and return a deque contains all images
    # and corresponding labels.
    try:
        with open('test_set_list', 'r') as f:
            eval_set_list = pickle.load(f)
        print('Eval set size: ' + str(len(eval_set_list)))
    except:
        raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')

    eval_set_queue = deque(eval_set_list)

    return eval_set_queue

def test():
    # load eval set queue.
    eval_set_queue = generate_eval_set()

    # build the tensorflow graph.
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

            result_list = []
            stats = {}
            stats['r'] = [0, 0, 0]  # [TP, FP, FN] for residential.
            stats['d'] = [0, 0, 0]  # [TP, FP, FN] for downtown/commercial.

            # initialize the result
            for ind in xrange(1, 66):
                result_list.append([ind, 0, 0, 0, 0]) #[region_index, TP, TN, FP, FN]

            for step in xrange(1, 936):
                start_time = time.time()
                # load data
                minibatch = []
                for count in xrange(0, BATCH_SIZE):
                    element = eval_set_queue.pop()
                    minibatch.append(element)

                image_list = [load_image(d[0]) for d in minibatch]
                label_list = [d[1] for d in minibatch]
                index_list = [d[2] for d in minibatch]
                type_list = [d[4] for d in minibatch]

                image_batch = np.array(image_list)

                image_batch = np.reshape(image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

                score = sess.run(logits, feed_dict={img_placeholder: image_batch})

                pos_score = np.exp(score[:, 1])/(np.exp(score[:, 1])+np.exp(score[:, 0]))

                for i in xrange(BATCH_SIZE):
                    if label_list[i][0] == 1 and pos_score[i] >= THRESHOLD: #TP
                        result_list[index_list[i]-1][1] += 1
                        stats[type_list[i]][0] += 1

                    elif label_list[i][0] == 1 and pos_score[i] < THRESHOLD: # FN
                        result_list[index_list[i]-1][4] += 1
                        stats[type_list[i]][2] += 1

                    elif label_list[i][0] == 0 and pos_score[i] < THRESHOLD: # TN
                        result_list[index_list[i]-1][2] += 1

                    elif label_list[i][0] == 0 and pos_score[i] >= THRESHOLD: # FP
                        result_list[index_list[i]-1][3] += 1
                        stats[type_list[i]][1] += 1

                duration = time.time() - start_time

                print("Batch " + str(step) + ", Duration: " + str(duration)+ "s, # images left: " + str(len(eval_set_queue)))

            # write csv
            with open(os.path.join("eval_result.csv"), 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['region', 'TP', 'TN', 'FP', 'FN'])
                writer.writerows(result_list)
            f.close()

            # print precision and recall.
            precision_r = float(stats['r'][0])/float(stats['r'][0] + stats['r'][1] + 0.00000001)
            recall_r = float(stats['r'][0])/float(stats['r'][0] + stats['r'][2] + + 0.00000001)

            precision_d = float(stats['d'][0]) / float(stats['d'][0] + stats['d'][1] + 0.00000001)
            recall_d = float(stats['d'][0]) / float(stats['d'][0] + stats['d'][2] + + 0.00000001)

            print ('############ RESULTS ############')
            print ('Residential: precision: ' + str(precision_r) + ' recall: '+str(recall_r))
            print ('Commercial: precision: ' + str(precision_d) + ' recall: ' + str(recall_d))
            print ('See region level analysis in eval_result.csv')

if __name__ == '__main__':
    test()
