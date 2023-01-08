import pytest
import tensorflow as tf
from data.preprocess.image import Resize


@pytest.mark.parametrize("input_size, output_size", [
    (  # test case
        (28, 28), (224, 224),
    ),
    (  # test case
        (1024, 1024), (256, 256),
    ),
])
def test_resize_cls_label(input_size, output_size):
    input_image = tf.zeros(shape=input_size+(3,), dtype=tf.float32)
    input_label = tf.zeros(shape=(), dtype=tf.int64)
    output_image, output_label = Resize(size=output_size)(input_image, input_label)
    assert output_image.shape == output_size + (3,)
    assert output_label == input_label


@pytest.mark.parametrize("input_size, output_size", [
    (  # test case
        (28, 28), (224, 224),
    ),
    (  # test case
        (1024, 1024), (256, 256),
    ),
])
def test_resize_seg_label(input_size, output_size):
    input_image = tf.zeros(shape=input_size+(3,), dtype=tf.float32)
    input_label = tf.zeros(shape=input_size, dtype=tf.int64)
    output_image, output_label = Resize(size=output_size)(input_image, input_label)
    assert output_image.shape == output_size + (3,)
    assert output_label.shape == output_size
