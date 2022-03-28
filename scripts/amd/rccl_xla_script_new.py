import os
import argparse
import numpy as np
import tensorflow as tf


# DISABLE_EAGER_EXECUTION = True
DISABLE_EAGER_EXECUTION = False

if DISABLE_EAGER_EXECUTION:
    # NOTE: make sure to run session to launch kernels
    tf.compat.v1.disable_eager_execution()

# enable xla
tf.config.optimizer.set_jit(True)


def main(log_dir):

    # set strategy
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=["GPU:0", "GPU:1"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3)
        ])
        return model

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        # Set reduction to `none` so we can do the reduction afterwards and divide by
        # global batch size.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=strategy.num_replicas_in_sync)

        # create model
        model = create_model()
        optimizer = tf.keras.optimizers.Adam()

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)

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
            tf.summary.scalar("loss", loss, step=step)
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=train_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    main(args.log_dir)
