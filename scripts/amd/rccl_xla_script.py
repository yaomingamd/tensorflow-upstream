import os
import argparse
import tensorflow as tf

# enable xla
# this works but partitions graph into xla and non xla
# tf.config.optimizer.set_jit(True)


def main(log_dir):

    # set strategy
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=["GPU:0", "GPU:1"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        weights = tf.Variable([[1.]])

        @tf.function(jit_compile=True) # force xla in tensorflow function
        def reduce_fn():
            ctx = tf.distribute.get_replica_context()
            input = tf.constant([[1.]])
            matmul_output = tf.linalg.matmul(input, weights)
            output = tf.math.add(matmul_output, tf.constant([[2.]]))
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
