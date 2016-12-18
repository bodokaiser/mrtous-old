import tensorflow as tf

import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1000,
    """Number of images to process per batch.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of threads to use to utilize in batch queue.""")
tf.app.flags.DEFINE_integer('num_epochs', 12,
    """Number of epochs to complete training.""")

tf.app.flags.DEFINE_string('data_dir', 'th-30',
    """Path to tfrecord files.""")
tf.app.flags.DEFINE_string('var_dir', '/tmp/mrtous',
    """Path to write var to.""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/mrtous',
    """Path to write logs to.""")

def main(argv):
    cnn = model.BasicCNN([13], data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_threads, num_epochs=FLAGS.num_epochs)

    mr, us = cnn.placeholder()
    us_ = cnn.interference(mr)

    loss = cnn.loss(us_, us)
    batch = cnn.batch()

    tf.summary.image('us', batch[1], max_outputs=1)
    tf.summary.image('us_rendered', us_, max_outputs=1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_path=FLAGS.var_dir+'/var.ckpt')

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        merged = tf.summary.merge_all()

        try:
            step = 0

            while not coord.should_stop():
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                rendered, norm, summary = sess.run([us_, loss, merged],
                    feed_dict={mr: batch[0].eval(), us: batch[1].eval()},
                    options=run_options, run_metadata=run_metadata)

                print('step: {}, norm: {:.0f}'.format(step, norm))

                writer.add_run_metadata(run_metadata, 'step{}'.format(step))
                writer.add_summary(summary, step)

                step += 1

        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            writer.close()

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()