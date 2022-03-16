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
    # input
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[1:2, ..., None]
    train_labels = train_labels[1:2, ...]

    # Getting the images in [0, 1] range.
    train_images = train_images / np.float32(255)

    # get outputs
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce(), devices=["GPU:0", "GPU:1"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # constants
    BUFFER_SIZE = len(train_images)
    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    EPOCHS = 1

    # data
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(
        train_dataset)

    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
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
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

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

        for epoch in range(EPOCHS):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            for x in train_dist_dataset:
                # print("x", x)
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            template = "Epoch {}, Loss: {}"
            print(template.format(epoch+1, train_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    main(args.log_dir)
