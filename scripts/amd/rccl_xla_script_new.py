import os
import argparse
from pickletools import optimize
import numpy as np
import tensorflow as tf


# enable xla
tf.config.optimizer.set_jit(True)


def main(log_dir):

    # set strategy
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=["GPU:0", "GPU:1"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():

        # create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2)
        ])
        optimizer = tf.keras.optimizers.SGD()

        def train_step(inputs):
            images, labels = inputs
            loss_object = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.nn.compute_average_loss(loss_object(
                    labels, predictions), global_batch_size=strategy.num_replicas_in_sync)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            return loss

        # `run` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(
                train_step, args=(dataset_inputs,))
            # print("per_replica_losses", per_replica_losses)
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        train_log_dir = os.path.join(log_dir, "ckpts")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Bracket the function call with
        tf.summary.trace_on(graph=True, profiler=True)
        dataset_inputs = tf.constant(
            [[1.0, 2.0, 3.0, 4.0, 5.0]]), tf.constant([2.0])
        loss = distributed_train_step(dataset_inputs)
        step = 1
        print("Step {}, Loss: {}".format(step, loss))

        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=step,
                profiler_outdir=train_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    main(args.log_dir)
