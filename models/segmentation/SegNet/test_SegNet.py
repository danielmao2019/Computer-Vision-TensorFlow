import pytest
import tensorflow as tf
from models.segmentation import SegNet

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)


@pytest.mark.dependency()
@pytest.mark.parametrize("input_shape", [
    (224, 224, 1), (224, 224, 3), (512, 512, 3),
])
def test_forward_pass(input_shape):
    model = SegNet(num_classes=10)
    model = model.build(input_shape=input_shape)
    input = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:])
    logging.debug(f"{input.shape=}")
    output = model(input)
    assert len(output.shape) == 4, f"{len(output.shape)=}"
    assert output.shape[:3] == input.shape[:3], f"{output.shape=}"


@pytest.mark.dependency(depends=['test_forward_pass'])
@pytest.mark.slow
@pytest.mark.parametrize("input_shape", [
    (128, 128, 3),
])
def test_overfit(input_shape):
    model = SegNet(num_classes=10)
    model = model.build(input_shape=input_shape)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=1.0e-01, momentum=0.9),
    )
    #x_train = tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:])
    #y_train = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:3], dtype=tf.int64)
    x_train = tf.concat([
    tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:]),
    #tf.ones(shape=(1,)+model.layers[0].input_shape[0][1:], dtype=tf.float32),
    tf.random.normal(shape=(1,)+model.layers[0].input_shape[0][1:]),
], axis=0)
    y_train = tf.concat([
    tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:3], dtype=tf.int64),
    #tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:3], dtype=tf.int64),
    tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:3], dtype=tf.int64),
], axis=0)
    logging.debug(f"{x_train.shape=}")
    logging.debug(f"{y_train.shape=}")
    model.trainable = True
    model.fit(x_train, y_train, epochs=50)
    model.trainable = False
    error = loss(y_true=y_train, y_pred=model(x_train))
    assert error <= 1.0e-5, f"{error=}"
