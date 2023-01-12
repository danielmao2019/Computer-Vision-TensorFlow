import pytest
from AlexNet import AlexNet
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)


def test_forward_pass(model):
    logging.info(f"[RUNNING] Test case 'forward_pass' started.")
    input = tf.zeros(shape=(1,)+model.layers[0].input_shape[0][1:])
    output = model(input)
    assert len(output.shape) == 2, f"{output.shape=}"
    assert output.shape[0] == input.shape[0], f"{output.shape=}"
    logging.info("[PASSED] Test case 'forward_pass' passed.")


def test_overfit(model):
    logging.info(f"[RUNNING] Test case 'overfit' started.")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=1.0e-03, momentum=0.9),
    )
    x_train = tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:])
    y_train = tf.zeros(shape=(1,), dtype=tf.int64)
    logging.debug(f"{x_train.shape=}")
    logging.debug(f"{y_train.shape=}")
    model.trainable = True
    model.fit(x_train, y_train, epochs=30)


if __name__ == "__main__":
    # num_classes = 10
    #case 1 
    model = AlexNet(num_classes=10)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=10)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=10)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 9
    #case 1 
    model = AlexNet(num_classes=9)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=9)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=9)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 8
    #case 1 
    model = AlexNet(num_classes=8)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=8)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=8)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 7
    #case 1 
    model = AlexNet(num_classes=7)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=7)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=7)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 6
    #case 1 
    model = AlexNet(num_classes=6)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=6)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=6)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 5
    #case 1 
    model = AlexNet(num_classes=5)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=5)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=5)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 4
    #case 1 
    model = AlexNet(num_classes=4)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=4)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=4)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 3
    #case 1 
    model = AlexNet(num_classes=3)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=3)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=3)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    #num_Classes = 2
    #case 1 
    model = AlexNet(num_classes=2)
    model = model.build(input_shape=(128, 128, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 2 
    model = AlexNet(num_classes=2)
    model = model.build(input_shape=(225, 225, 3))
    test_forward_pass(model)
    test_overfit(model)
    #case 3
    model = AlexNet(num_classes=2)
    model = model.build(input_shape=(128, 128, 1))
    test_forward_pass(model)
    test_overfit(model)
    