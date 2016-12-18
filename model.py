import tensorflow as tf

import os
import os.path

PATCH_SIZE = 7
PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE

def _format_filenames(data_dir, records):
    fn = lambda r: os.path.join(data_dir, '{:02}.tfrecord'.format(r))
    return list(map(fn, records))

def _count_patches(filenames):
    fn = lambda f: sum(1 for _ in tf.python_io.tf_record_iterator(f))
    return list(map(fn, filenames))

def _conv_layer(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

def _weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape), name='weight')

def _bias_variable(shape):
    return tf.Variable(tf.constant(.1, shape=shape), name='bias')

class BasicCNN(object):

    def __init__(self, records, data_dir, batch_size, num_threads, num_epochs):
        filenames = _format_filenames(data_dir, records)

        self._filenames = filenames
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._num_threads = num_threads
        self._num_patches = _count_patches(filenames)

    def batch(self):
        queue = tf.train.string_input_producer(self._filenames,
            num_epochs=self.num_epochs)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader()
            _, example = reader.read(queue)

        with tf.name_scope('features'):
            features = tf.parse_single_example(example, features={
                'us_patch': tf.FixedLenFeature([], tf.string),
                'mr_patch': tf.FixedLenFeature([], tf.string),
            })

        with tf.name_scope('decode'):
            mr_raw = tf.decode_raw(features['mr_patch'], tf.float64)
            mr = tf.cast(tf.reshape(mr_raw, [PATCH_SIZE, PATCH_SIZE, 1]), tf.float32)
            us_raw = tf.decode_raw(features['us_patch'], tf.float64)
            us = tf.cast(tf.reshape(us_raw, [PATCH_SIZE, PATCH_SIZE, 1]), tf.float32)

        return tf.train.batch([mr, us],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            capacity=3*self.batch_size)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_patches(self):
        return np.sum(self._num_patches)

    @property
    def num_threads(self):
        return self._num_threads

    def placeholder(self):
        with tf.name_scope('placeholder'):
            x = tf.placeholder(tf.float32, [None, PATCH_SIZE, PATCH_SIZE, 1])
            y = tf.placeholder(tf.float32, [None, PATCH_SIZE, PATCH_SIZE, 1])

        return x, y

    def interference(self, x):
        with tf.name_scope('conv1'):
            conv1_w = _weight_variable([3, 3, 1, 3])
            conv1_b = _bias_variable([3])
            conv1 = _conv_layer(x, conv1_w) + conv1_b

            for i in range(3):
                tf.summary.scalar('bias{}'.format(i), conv1_b[i])
                tf.summary.image('filter{}'.format(i),
                    tf.expand_dims(conv1_w[:, :, :, i], 0),
                    max_outputs=1)

        with tf.name_scope('conv2'):
            conv2_w = _weight_variable([1, 1, 3, 1])
            conv2 = _conv_layer(conv1, conv2_w)

            for i in range(3):
                tf.summary.scalar('param{}'.format(i), conv2_w[0, 0, i, 0])

        return conv2

    def loss(self, f, y):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(f-y)

        return loss

    def training(self, loss):
        tf.summary.scalar('loss', loss)

        return tf.train.AdamOptimizer().minimize(loss)