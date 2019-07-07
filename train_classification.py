#!/usr/bin/env python
"""Train the inception-v3 model on Solar Panel Identification dataset."""


from datetime import datetime
import os.path
import time
import sys

import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import random
import pickle
from collections import deque

from inception import  inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_save_dir', 'ckpt/inception_classification',
                           """Directory for saving model checkpoint. """)

tf.app.flags.DEFINE_string('ckpt_restore_dir', 'ckpt/inception_classification',
                           """Directory for restoring old model checkpoint. """)

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 'ckpt/inception-v3/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('train_set_dir', 'SPI_train',
                           """Directory of training set""")

tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches/steps to run.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('fine_tune', True,
                            """If true, start from well-trained model on SPI dataset, else start from
                            pretrained model on ImageNet""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")

# basic parameters
BATCH_SIZE = 32
IMAGE_SIZE = 299
NUM_CLASSES = 2

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 0.1              # Epsilon term for RMSProp.

def load_image(path):
    # load image and prepocess.
    rotate_angle_list = [0, 90, 180, 270]
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    rotate_angle = random.choice(rotate_angle_list)
    image = skimage.transform.rotate(resized_img, rotate_angle)
    return image

def train():
    # load train set list and transform it to queue.
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

    # build the tensorflow graph.
    with tf.Graph().as_default() as g:

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = train_set_size / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)

        images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = inception.inference(images, NUM_CLASSES, for_training=True,
                                     restore_logits=FLAGS.fine_tune,
                                     scope=None)

        inception.loss(logits, labels, batch_size=BATCH_SIZE)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope = None)

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope=None)

        # Calculate the gradients for the batch of data on this ImageNet
        # tower.
        grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY, global_step)

        variables_to_average = (tf.trainable_variables() +
                                tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # open session and initialize
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))
        sess.run(init)

        # restore old checkpoint
        if FLAGS.fine_tune:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
        else:
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        summary_writer = tf.summary.FileWriter(
            FLAGS.ckpt_save_dir,
            graph_def=sess.graph.as_graph_def(add_shapes=True))

        step = 1
        while step <= FLAGS.max_steps:
            start_time = time.time()
            # construct image batch and label batch for one step train
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

            _, loss_value = sess.run([train_op, total_loss], feed_dict={images: image_batch, labels: label_batch})

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

            # shuttle the image list per epoch
            if step % num_batches_per_epoch == 0:
                random.shuffle(train_set_queue)

            # write summary periodically
            if step == 1 or step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: image_batch, labels: label_batch})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.ckpt_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1


if __name__ == '__main__':
    train()
