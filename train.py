import numpy as np
import tensorflow as tf

import time
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1000,
    """Number of images to process per batch.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of threads to use to utilize in batch queue.""")
tf.app.flags.DEFINE_integer('num_epochs', 32,
    """Number of epochs to complete training.""")

tf.app.flags.DEFINE_string('data_dir', 'th-30',
    """Path to tfrecord files.""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/mrtous',
    """Path to write logs to.""")

def main(args):
    cnn = model.BasicCNN([13], data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads)

    mr, us = cnn.placeholder()
    us_ = cnn.interference(mr)

    loss = cnn.loss(us_, us)
    train = cnn.training(loss, .0001)
    batch = cnn.batch()
    iters = cnn.iterations()

    print('iters', iters)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        merged = tf.summary.merge_all()

        try:
            step = 0
            epoch = 0

            while not coord.should_stop() or epoch < FLAGS.num_epochs:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, summary = sess.run([train, merged], feed_dict={
                    mr: batch[0].eval(),
                    us: batch[1].eval(),
                }, options=run_options, run_metadata=run_metadata)

                print('epoch: {}, iteration: {}'.format(epoch, step))
                writer.add_run_metadata(run_metadata, 'epoch{}step{}'.format(epoch, step))
                writer.add_summary(summary, step)

                if step > iters:
                    step = 0
                    epoch += 1
                else:
                    step += 1

        except tf.errors.OutOfRangeError:
            print('done with training')
        finally:
            writer.close()

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()