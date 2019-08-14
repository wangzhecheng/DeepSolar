#!/usr/bin/env python
#
# Save Inception-v3 model on test(eval) set.
# Usage
#   python save_model.py
#

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
        img_placeholder = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
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
