import os
import argparse
from pickletools import optimize
import numpy as np
import tensorflow as tf


# enable xla
tf.config.optimizer.set_jit(True)


def main(log_dir):

    # set strategy
    gpus = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"][:2]
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=gpus)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        # create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2)
        ])
        optimizer = tf.keras.optimizers.SGD()

        # train function
        @tf.function
        def distributed_train_step(dataset_inputs):
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

            per_replica_losses = strategy.run(
                train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                   axis=None)

        # run train function
        train_log_dir = os.path.join(log_dir, "ckpts")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        dataset_inputs = tf.constant(
            [[1.0]]), tf.constant([2.0])
        tf.summary.trace_on(graph=True, profiler=True)
        loss = distributed_train_step(dataset_inputs)
        step = 1

        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=step,
                profiler_outdir=train_log_dir)

        print("Step {}, Loss: {}".format(step, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    main(args.log_dir)
