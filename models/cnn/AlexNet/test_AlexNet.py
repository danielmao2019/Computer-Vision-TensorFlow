from AlexNet import AlexNet
import tensorflow as tf


def test_training(model, image_h, image_w):
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=1.0e-03),
    )
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, axis=3)
    x_train = tf.image.resize(x_train, size=(image_h, image_w))
    print(f"{x_train.shape=}, {y_train.shape=}")
    model.fit(x_train, y_train, epochs=1)


def test_forward_pass(model, image_h, image_w):
    batch_size = 8
    fake_input = tf.zeros(shape=(1, image_h, image_w, 1))
    fake_output = model(fake_input)
    print(f"{fake_output.shape=}")


if __name__ == "__main__":
    image_h = image_w = 128
    model = AlexNet(num_classes=10)
    model = model.build(input_shape=(image_h, image_w, 1))
    test_forward_pass(model, image_h, image_w)
    test_training(model, image_h, image_w)
