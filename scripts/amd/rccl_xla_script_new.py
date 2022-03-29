import os
import argparse
from pickletools import optimize
import numpy as np
import tensorflow as tf


# DISABLE_EAGER_EXECUTION = True
# # DISABLE_EAGER_EXECUTION = False

# if DISABLE_EAGER_EXECUTION:
#     # NOTE: make sure to run session to launch kernels
#     tf.compat.v1.disable_eager_execution()

# enable xla
tf.config.optimizer.set_jit(True)


def main(log_dir):

    # set strategy
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=["GPU:0", "GPU:1"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        dense = tf.keras.layers.Dense(2)
        @tf.function
        def reduce_fn():
            ctx = tf.distribute.get_replica_context()
            output = dense(tf.constant([[1.]]))
            return ctx.all_reduce(tf.distribute.ReduceOp.SUM, output)

        # run train function
        train_log_dir = os.path.join(log_dir, "ckpts")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        tf.summary.trace_on(graph=True, profiler=True)
        result = strategy.run(reduce_fn)
        print(result)
        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=1,
                profiler_outdir=train_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    main(args.log_dir)
