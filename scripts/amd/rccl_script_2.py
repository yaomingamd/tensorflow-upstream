import tensorflow as tf
import numpy as np

import os
import argparse

from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import collective_util


def make_collective(num_processes, gpu_per_process):
    """Returns collectives and other info to be used in tests.

    Args:
      num_processes: an integer indicating the number of processes that
        participate in the collective.
      gpu_per_process: number of GPUs (0 if no GPUs) used by each process.

    Returns:
     A tuple of (collective, devices, pid) where collective is a instance
     of `CollectiveAllReduce`, devices are a list of local devices (str)
     attached to the current process, and pid is the id of this process among
     all participant processes.
    """

    # cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()  # not needed locally
    # task_id=cluster_resolver.task_id
    task_id=0
    devices = [
        "/job:localhost/replica:0/task:%d/device:CPU:0" % task_id
    ]
    if gpu_per_process > 0:
        devices = [
            "/job:localhost/replica:0/task:%d/device:GPU:%d" %
            (task_id, i) for i in range(gpu_per_process)
        ]
    group_size = num_processes * len(devices)
    collective = cross_device_ops_lib.CollectiveAllReduce(
        devices=devices,
        group_size=group_size,
        options=collective_util.Options())
    return collective, devices, task_id


collective, devices, pid = make_collective(1, 2)
print(collective, devices, pid)


# enable xla
# tf.config.optimizer.set_jit(True)


# pick distributed strategy
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce(), devices=["/gpu:0", "/gpu:1", "/gpu:2"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    tf.compat.v1.disable_eager_execution()

    # cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    size = 1000000000
    x = tf.compat.v1.placeholder(tf.float32, size)

    with tf.device("/gpu:0"):
        first_batch = tf.slice(x, [0], [size//2])
        mean1 = tf.reduce_sum(first_batch)
        # mean1 = tf.raw_ops.NcclReduce(input=[first_batch], reduction="sum")

    with tf.device("/gpu:1"):
        second_batch = tf.slice(x, [size//2], [-1])
        mean2 = tf.reduce_sum(second_batch)
        # mean = tf.raw_ops.NcclReduce(
        #     input=[first_batch, second_batch], reduction="sum")
        # mean = tf.raw_ops.NcclAllReduce(
        #     input=[first_batch, second_batch], reduction="sum", num_devices=2, shared_name="ncclallreduce")
        #

    with tf.device("/gpu:2"):
        mean = (mean1 + mean2) / 2
        # mean = tf.reduce_mean([first_batch, second_batch])

    print(first_batch.device, first_batch)
    print(second_batch.device, second_batch)
    print(mean.device, mean)

    with tf.compat.v1.Session() as sess:
        result = sess.run(mean, feed_dict={x: np.random.random(size)})
        print(result)
