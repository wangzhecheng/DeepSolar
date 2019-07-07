#!/usr/bin/env python
"""Generate Class Activation Map for positive samples in """
import sys
import copy
import os.path
import re
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import pickle
import csv
from collections import deque

# from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('classification_ckpt_restore_dir', 'inception_classification',
                           """ Directory for restoring parameters of classification model. """)

tf.app.flags.DEFINE_string('segmentation_ckpt_restore_dir', 'inception_segmentation',
                           """ Directory for restoring parameters of segmentation branch. """)

tf.app.flags.DEFINE_string('eval_set_dir', 'SPI_eval',
                           """ Directory of test set. """)


# basic parameters
BATCH_SIZE = 1
IMAGE_SIZE = 299
NUM_CLASSES = 2
SEGMENTATION_THRES = 0.37 # threshold for segmenting solar panel
RESULT_DIR = 'segmentation_results'

def load_image(path):
    # load and prepocess image
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    return resized_img

def rescale_CAM(classmap_val):
    # rescale class activation map to [0, 1].
    CAM_rescale = map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_val)
    CAM_rescale = CAM_rescale[0]
    return CAM_rescale

def generate_eval_set():
    # load all train data and return a deque contains all images
    # and corresponding labels.
    try:
        with open('test_set_list.pickle', 'r') as f:
            eval_set_list = pickle.load(f)
        print('Eval set size: ' + str(len(eval_set_list)))
    except:
        raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')

    eval_set_queue = deque(eval_set_list)

    return eval_set_queue

def test():
    eval_set_queue = generate_eval_set()

    with tf.Graph().as_default() as g:
        img_placeholder = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3])

        logits, _, feature_map = inception.inference(img_placeholder, NUM_CLASSES)


        with tf.name_scope('conv_aux_1') as scope:
            kernel1 = tf.Variable(tf.truncated_normal([3, 3, 288, 512], dtype=tf.float32, stddev=1e-4), name='weights')
            conv = tf.nn.conv2d(feature_map, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases1)
            conv_aux = tf.nn.relu(bias, name=scope)

        with tf.name_scope('conv_aux_2') as scope:
            kernel2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-4), name='weights')
            conv = tf.nn.conv2d(conv_aux, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases2)
            conv_aux = tf.nn.relu(bias, name=scope)

        GAP = tf.reduce_mean(conv_aux, [1, 2])

        W = tf.get_variable(name='W', shape=[512, 2], initializer=tf.random_normal_initializer(0., 0.01))

        conv_map_resized = tf.image.resize_bilinear(conv_aux, [100, 100])

        # get weights connected to definite class.
        W_c = tf.gather(tf.transpose(W), 1)
        W_c = tf.reshape(W_c, [-1, 512, 1])
        conv_map_resized = tf.reshape(conv_map_resized, [-1, 100 * 100, 512])
        CAM = tf.batch_matmul(conv_map_resized, W_c)
        CAM = tf.reshape(CAM, [-1, 100, 100])

        # Construct saver
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        print variables_to_restore
        saver1 = tf.train.Saver(variables_to_restore)
        saver2 = tf.train.Saver(var_list=[W, kernel2, biases2, kernel1, biases1])

        with tf.Session() as sess:
            # restore model parameters.
            checkpoint1 = tf.train.get_checkpoint_state(FLAGS.classification_ckpt_restore_dir)
            if checkpoint1 and checkpoint1.model_checkpoint_path:
                saver1.restore(sess, checkpoint1.model_checkpoint_path)
                print("Successfully loaded:", checkpoint1.model_checkpoint_path)
            else:
                print("Could not find old network weights")

            checkpoint2 = tf.train.get_checkpoint_state(FLAGS.segmentation_ckpt_restore_dir)
            if checkpoint2 and checkpoint2.model_checkpoint_path:
                saver2.restore(sess, checkpoint2.model_checkpoint_path)
                print("Successfully loaded:", checkpoint2.model_checkpoint_path)
            else:
                print("Could not find old network weights")

            stats = {}
            stats['r'] = [0, 0, 0]  # [TP, FP, FN] for residential.
            stats['d'] = [0, 0, 0]  # [TP, FP, FN] for downtown/commercial.
            area_error = {}
            area_error['r'] = []
            area_error['d'] = []

            # store both true and estimate total pixel areas for each region
            true_total_area = {}
            for i in xrange(1, 66):
                true_total_area[i] = 0.0
            estimiate_total_area = {}
            for i in xrange(1, 66):
                estimiate_total_area[i] = 0.0

            for step in xrange(1, len(eval_set_queue)+1):
                print ('Processing '+str(step)+'/'+str(len(eval_set_queue))+'...')
                img_path, label, region_index, img_index, region_type = eval_set_queue.pop()
                img = load_image(img_path)
                img_batch = np.reshape(img, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
                score = sess.run(logits, feed_dict={img_placeholder: img_batch})
                pos_prob = np.exp(score[0, 1]) / (np.exp(score[0, 1]) + np.exp(score[0, 0]))

                if pos_prob >= 0.5:
                    # generate CAM for that sample
                    CAM_val = sess.run(CAM, feed_dict={img_placeholder: img_batch})
                    CAM_val = rescale_CAM(CAM_val)
                    pred_pixel_area = np.sum(CAM_val > SEGMENTATION_THRES) # predicted/estimated pixel area
                    estimiate_total_area[region_index] += pred_pixel_area

                    if label == [0]: # FP
                        stats[region_type][1] += 1
                        # save original image and CAM.
                        skimage.io.imsave(os.path.join(RESULT_DIR, 'FP', str(region_index) + '_' + str(img_index) + '_original.png'), img)
                        skimage.io.imsave(os.path.join(RESULT_DIR, 'FP', str(region_index) + '_' + str(img_index) + '_CAM.png'), img)

                    else: # TP
                        stats[region_type][0] += 1
                        # save original image and CAM.
                        skimage.io.imsave(os.path.join(RESULT_DIR, 'TP', str(region_index) + '_' + str(img_index) + '_original.png'),img)
                        skimage.io.imsave(os.path.join(RESULT_DIR, 'TP', str(region_index) + '_' + str(img_index) + '_CAM.png'), img)
                        # compare with ground truth segmentation.
                        true_seg_img = skimage.io.imread(os.path.join(FLAGS.eval_set_dir, str(region_index), str(img_index)+'_true_seg.png'))
                        true_seg_img /= 255.0
                        true_pixel_area = np.sum(true_seg_img)
                        true_pixel_area = true_pixel_area * (100 * 100) / (320 * 320)
                        true_total_area[region_index] += true_pixel_area
                        area_error[region_type].append(true_pixel_area - pred_pixel_area)

                else:
                    if label == [1]:  # FN
                        stats[region_type][2] += 1
                        true_seg_img = skimage.io.imread(
                            os.path.join(FLAGS.eval_set_dir, str(region_index), str(img_index) + '_true_seg.png'))
                        true_seg_img /= 255.0
                        true_pixel_area = np.sum(true_seg_img)
                        true_pixel_area = true_pixel_area * (100 * 100) / (320 * 320)
                        true_total_area[region_index] += true_pixel_area

            # report precision and recall and absolute error rate.
            abs_error_sum_r = 0
            for e in area_error['r']:
                abs_error_sum_r += abs(e)
            abs_error_rate_r = float(abs_error_sum_r)/float(len(area_error['r']))

            abs_error_sum_d = 0
            for e in area_error['d']:
                abs_error_sum_d += abs(e)
            abs_error_rate_d = float(abs_error_sum_d) / float(len(area_error['d']))

            precision_r = float(stats['r'][0]) / float(stats['r'][0] + stats['r'][1] + 0.00000001)
            recall_r = float(stats['r'][0]) / float(stats['r'][0] + stats['r'][2] + + 0.00000001)

            precision_d = float(stats['d'][0]) / float(stats['d'][0] + stats['d'][1] + 0.00000001)
            recall_d = float(stats['d'][0]) / float(stats['d'][0] + stats['d'][2] + + 0.00000001)

            print ('############ RESULTS ############')
            print ('Residential: precision: ' + str(precision_r) + ' recall: ' + str(recall_r) +
                   ' average absolute error rate: ' + str(abs_error_rate_r))
            print ('Commercial: precision: ' + str(precision_d) + ' recall: ' + str(recall_d) +
                   ' average absolute error rate: ' + str(abs_error_rate_d))

            # save csv for region-level comparison of true total area and estimated total area.
            result_list = []
            for i in xrange(1, 66):
                result_list.append([i, true_total_area[i], estimiate_total_area[i],
                                   float(estimiate_total_area[i] - true_total_area[i])/float(true_total_area[i])])
            with open(os.path.join("region_level_area_estimation.csv"), 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['region', 'true pixel area', 'estimiated pixel area', 'relative difference'])
                writer.writerows(result_list)
            f.close()

            
if __name__ == '__main__':
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
        os.mkdir(os.path.join(RESULT_DIR, 'TP'))
        os.mkdir(os.path.join(RESULT_DIR, 'FP'))
    test()
