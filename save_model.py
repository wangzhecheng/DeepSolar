#!/usr/bin/env python
#
# Save Inception-v3 model on test(eval) set.
# Usage
#   python save_model.py
#


""""

Saves the classification model in the tensorflow SavedModel format. 
See https://www.tensorflow.org/guide/saved_model

The graph is saved such that it expects an image of shape [299,299,3]
consisting of uint8 values in the range [0,255]
"""

from inception import inception_model as inception
import tensorflow as tf
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS
checkpoint_bucket = 'gs://solarweb/deepsolar/inception_classification'
tf.app.flags.DEFINE_string('ckpt_dir', checkpoint_bucket,
                           "Directory for restoring trained model checkpoints.")

IMAGE_SIZE = 299
NUM_CLASSES = 2


def main():

    # build the tensorflow graph.
    with tf.Graph().as_default() as g:
        input_shape = [IMAGE_SIZE, IMAGE_SIZE, 3]
        final_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
        img_placeholder = tf.placeholder(
            tf.uint8, shape=input_shape)
        print(img_placeholder.shape)
        # reshape to add batch size dimension
        img = tf.reshape(img_placeholder, final_shape)
        # cast to float
        img = tf.dtypes.cast(img, dtype=tf.float32)
        # normalize input to values in range [0,1]
        img = img / 255.0
        print('Image shape {}'.format(img.shape))
        logits, _ = inception.inference(img, NUM_CLASSES)
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
            save_dir = './saved_models5'
            print(
                'Saving model for deployment in directory {}'.format(save_dir))
            tf.saved_model.simple_save(sess,
                                       save_dir,
                                       inputs={
                                           'image': img_placeholder},
                                       outputs={'predictions': logits})


if __name__ == '__main__':
    main()
