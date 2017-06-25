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

from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_save_dir', 'ckpt/inception_classification',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'SPI_train',
                           """Directory of training set""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', True,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'ckpt/pretrained_inception/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")

# basic parameters
BATCH_SIZE = 32
IMAGE_SIZE = 299
NUM_CLASSES = 2
# training sample range
RANGE = 30000
# number of training samples
TRAIN_SET_SIZE = 37500
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
    data_dir_2 = os.path.join(FLAGS.data_dir, 'old/0_2')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_2, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1
    data_dir_3 = os.path.join(FLAGS.data_dir, 'old/0_4')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_3, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1
    data_dir_4 = os.path.join(FLAGS.data_dir, 'old/0_5')
    for i in xrange(1, RANGE):
        f = os.path.join(data_dir_4, '%d.png' % i)
        if not os.path.exists(f):
            continue
        image_path.append((f, [0]))
        neg_num += 1
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
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = TRAIN_SET_SIZE / BATCH_SIZE
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
                                     restore_logits=False,
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
            log_device_placement=False))
        sess.run(init)

        # restore old checkpoint
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir,
            graph_def=sess.graph.as_graph_def(add_shapes=True))

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

            if step == 1 or step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: image_batch, labels: label_batch})
                summary_writer.add_summary(summary_str, step)
                print label_batch
                score = sess.run(logits, feed_dict={images: image_batch, labels: label_batch})
                print score[0]

            # Save the model checkpoint periodically.
            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # Give training error and validation error every 400 steps
            if step % 1000 == 0:
                print("begin calculating training error at step ", step)
                TP1 = TN1 = FP1 = FN1 = 0
                for ii in xrange(0, 50):
                    minibatch2 = random.sample(train_set, BATCH_SIZE)
                    image_list2 = [load_image(d[0]) for d in minibatch2]
                    label_list2 = [d[1] for d in minibatch2]
                    image_batch2 = np.array(image_list2)
                    label_batch2 = np.array(label_list2)
                    image_batch2 = np.reshape(image_batch2, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
                    label_batch2 = np.reshape(label_batch2, [BATCH_SIZE])
                    score = sess.run(logits, feed_dict={images: image_batch2})
                    for k in xrange(0, BATCH_SIZE):
                        if label_batch2[k] == 1 and score[0][k, 1] >= 0:
                            TP1 += 1
                        elif label_batch2[k] == 1 and score[0][k, 1] < 0:
                            FN1 += 1
                        elif label_batch2[k] == 0 and score[0][k, 1] <= 0:
                            TN1 += 1
                        elif label_batch2[k] == 0 and score[0][k, 1] > 0:
                            FP1 += 1
                precision = float(TP1) / float(TP1 + FP1)
                recall = float(TP1) / float(TP1 + FN1)
                print("Training set: TP:", TP1, "TN:", TN1, "FP:", FP1, "FN:", FN1, "precision:", precision, "recall:",
                      recall)
                train_record.append(('train', step, TP1, TN1, FP1, FN1, precision, recall))

                print("begin calculating validation error at step ", step)
                TP2 = TN2 = FP2 = FN2 = 0
                for ii in xrange(0, 50):
                    minibatch3 = random.sample(val_set, BATCH_SIZE)
                    image_list3 = [load_image(d[0]) for d in minibatch3]
                    label_list3 = [d[1] for d in minibatch3]
                    image_batch3 = np.array(image_list3)
                    label_batch3 = np.array(label_list3)
                    image_batch3 = np.reshape(image_batch3, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
                    label_batch3 = np.reshape(label_batch3, [BATCH_SIZE])
                    score = sess.run(logits, feed_dict={images: image_batch3})
                    for k in xrange(0, BATCH_SIZE):
                        if label_batch3[k] == 1 and score[0][k, 1] >= 0:
                            TP2 += 1
                        elif label_batch3[k] == 1 and score[0][k, 1] < 0:
                            FN2 += 1
                        elif label_batch3[k] == 0 and score[0][k, 1] <= 0:
                            TN2 += 1
                        elif label_batch3[k] == 0 and score[0][k, 1] > 0:
                            FP2 += 1
                precision2 = float(TP2) / float(TP2 + FP2)
                recall2 = float(TP2) / float(TP2 + FN2)
                print(
                "Validation set: TP:", TP2, "TN:", TN2, "FP:", FP2, "FN:", FN2, "precision:", precision2, "recall:",
                recall2)
                train_record.append(('val', step, TP2, TN2, FP2, FN2, precision2, recall2))

                with open("record.pickle", 'w') as f:
                    pickle.dump(train_record, f)

            step += 1



if __name__ == '__main__':
    train()



