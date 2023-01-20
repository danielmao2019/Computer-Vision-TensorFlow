from ResNet import ResNet
import tensorflow as tf
import pytest
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

@pytest.mark.dependency()
@pytest.mark.parametrize("input_shape", [
    (224, 224, 1), (224, 224, 3), (512, 512, 3),
])
def test_forward_pass(input_shape):
    image_h = image_w = 64
    model = ResNet(num_classes=10, version=18)
    model = model.build(input_shape=(image_h, image_w, 1))
    input = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:])
    logging.debug(f"{input.shape=}")
    output = model(input)
    assert len(output.shape) == 2, f"{len(output.shape)=}"
    assert output.shape[0] == input.shape[0], f"{output.shape=}"


@pytest.mark.dependency(depends=['test_forward_pass'])
@pytest.mark.slow
@pytest.mark.parametrize("input_shape", [
    (128, 128, 3),
])
def test_overfit(input_shape):
    image_h = image_w = 64
    model = ResNet(num_classes=10, version=18)
    model = model.build(input_shape=(image_h, image_w, 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        loss=loss,
        optimizer=tf.optimizers.SGD(),
    )
    x_train = tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:])
    y_train = tf.zeros(shape=(1,), dtype=tf.int64)
    logging.debug(f"{x_train.shape=}")
    logging.debug(f"{y_train.shape=}")
    model.trainable = True
    model.fit(x_train, y_train, epochs=1000)
    model.trainable = False
    error = loss(y_true=y_train, y_pred=model(x_train))
    assert error <= 1.0e-5, f"{error=}"


if __name__ == "__main__":
    image_h = image_w = 64
    model = ResNet(num_classes=10, version=18)
    model = model.build(input_shape=(image_h, image_w, 1))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(),
    )
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, axis=3)
    x_test = tf.expand_dims(x_test, axis=3)
    x_train = tf.image.resize(x_train, size=(image_h, image_w))
    x_test = tf.image.resize(x_test, size=(image_h, image_w))
    model.fit(x_train, y_train, epochs=1)
