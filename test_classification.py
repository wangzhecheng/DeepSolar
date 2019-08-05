#!/usr/bin/env python
#
# Evaluate Inception-v3 model on test(eval) set.
# Usage
#   find . -name *.jpg | test_classification.py <out.csv>
#

from inception.slim import slim
from inception import inception_model as inception
from collections import deque
import skimage.transform
import skimage.io
import skimage
import tensorflow as tf
import numpy as np
import csv
import pickle
import time
import re
import sys
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS
checkpoint_bucket = 'gs://solarweb/deepsolar/inception_classification'
tf.app.flags.DEFINE_string('ckpt_dir', checkpoint_bucket,
                           "Directory for restoring trained model checkpoints.")

BATCH_SIZE = 1
IMAGE_SIZE = 299
NUM_CLASSES = 2
SAVE_MODEL = True


def load_image(path):
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]

    dir, filename = os.path.split(path)

    skimage.io.imsave("%s/out_%s" % (dir, filename), resized_img)
    return resized_img


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():

    global BATCH_SIZE

    # List of all images to process
    filelist = [file.strip() for file in sys.stdin]

    if len(filelist) < BATCH_SIZE:
        BATCH_SIZE = len(filelist)

    # build the tensorflow graph.
    with tf.Graph().as_default() as g:
        img_placeholder = tf.placeholder(
            tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
        print(img_placeholder.shape)
        logits, _ = inception.inference(img_placeholder, NUM_CLASSES)
        saver = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))

        with sess:
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print('Restoring trained model from checkpoint')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint loaded')
            else:
                print('No checkpoint file found')

            for files_batch in chunks(filelist, BATCH_SIZE):

                start_time = time.time()

                image_list = [load_image(file) for file in files_batch]
                image_batch = np.array(image_list)
                print(image_batch.shape)
                image_batch = np.reshape(
                    image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

                score = sess.run(logits, feed_dict={
                                 img_placeholder: image_batch})

                pos_score = np.exp(
                    score[:, 1])/(np.exp(score[:, 1])+np.exp(score[:, 0]))

                for i in range(BATCH_SIZE):
                    print("Score %s : %f" % (files_batch[i], pos_score[i]))

                duration = time.time() - start_time

                print("Batch done Duration: " + str(duration))
            if SAVE_MODEL:
                try:
                    save_dir = './saved_models'
                    print(
                        'Saving model for deployment in directory {}'.format(save_dir))
                    tf.saved_model.simple_save(sess,
                                               save_dir,
                                               inputs={
                                                   'image': img_placeholder},
                                               outputs={'predictions': logits})
                except Exception as e:
                    print('Failed to save model withe exception {}'.format(e))


if __name__ == '__main__':
    main()
