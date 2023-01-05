from SegNet import SegNet
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)


def test_forward_pass(model):
    logging.info(f"[RUNNING] Test case 'forward_pass' started.")
    input = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:])
    logging.debug(f"{input.shape=}")
    output = model(input)
    assert len(output.shape) == 4, f"{len(output.shape)=}"
    assert output.shape[:3] == input.shape[:3], f"{output.shape=}"
    logging.info("[PASSED] Test case 'forward_pass' passed.")


def test_overfit(model):
    logging.info(f"[RUNNING] Test case 'overfit' started.")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=1.0e-01, momentum=0.9),
    )
    x_train = tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:])
    y_train = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:3], dtype=tf.int64)
    logging.debug(f"{x_train.shape=}")
    logging.debug(f"{y_train.shape=}")
    model.trainable = True
    model.fit(x_train, y_train, epochs=50)


if __name__ == "__main__":
    model = SegNet(num_classes=10)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
