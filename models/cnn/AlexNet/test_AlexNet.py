from AlexNet import AlexNet
import tensorflow as tf


if __name__ == "__main__":
    image_h = image_w = 128
    model = AlexNet(num_classes=10)
    model = model.build(input_shape=(image_h, image_w, 1))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=1.0e-03),
    )
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, axis=3)
    x_test = tf.expand_dims(x_test, axis=3)
    x_train = tf.image.resize(x_train, size=(image_h, image_w))
    x_test = tf.image.resize(x_test, size=(image_h, image_w))
    model.fit(x_train, y_train, epochs=1)
