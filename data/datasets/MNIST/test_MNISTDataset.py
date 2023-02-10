import matplotlib.pyplot as plt
import pytest
from data.datasets.MNIST.MNISTDataset import MNISTDataset
import tensorflow_datasets as tfds
import tensorflow as tf

# unit tests checking example from MNIST dataset for training asserting data shape, type, and load count
def test_MNIST_dataset():
    dataset = MNISTDataset(purpose='training')
    assert(len(dataset.core) == 60000), f"{len(dataset.core)=}"
    image, label = dataset.get_example()
    assert(len(image.shape) == 3), f"{len(image.shape)=}"
    assert(image.shape == (28, 28, 1)), f"{image.shape=}"
    assert(len(label.shape) == 0), f"{len(label.shape)=}"
    assert(image.dtype == tf.float32), f"{image.dtype=}"
    assert(label.dtype == tf.int64), f"{label.dtype=}"
