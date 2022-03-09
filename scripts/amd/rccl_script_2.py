import tensorflow as tf
import numpy as np

import os
import argparse

# enable xla
tf.config.optimizer.set_jit(True)

# pick distributed strategy
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce(), devices=["/gpu:0", "/gpu:1"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    tf.compat.v1.disable_eager_execution()

    # cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    
    x = tf.compat.v1.placeholder(tf.float32, 100)

    with tf.device("/gpu:0"):
        first_batch = tf.slice(x, [0], [50])
        mean1 = tf.reduce_mean(first_batch)

    with tf.device("/gpu:1"):
        second_batch = tf.slice(x, [50], [-1])
        mean2 = tf.reduce_mean(second_batch)
        mean = (mean1 + mean2) / 2

    print(mean1.device, mean1)
    print(mean2.device, mean2)
    print(mean.device, mean)

    with tf.compat.v1.Session() as sess:
        result = sess.run(mean, feed_dict={x: np.random.random(100)})
        print(result)
