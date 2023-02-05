import matplotlib.pyplot as plt
import pytest
# from MNISTDataset import MNISTDataset # use this import if running from Computer-Vision-TensorFlow/data/datasets/MNIST
from data.datasets.MNIST.MNISTDataset import MNISTDataset # use this import if running pytest from Computer-Vision-TensorFlow
import tensorflow_datasets as tfds
import tensorflow as tf

# unit tests checking example from MNIST dataset for training asserting data shape, type, and load count
def test_MNIST_dataset():
    dataset = MNISTDataset(purpose='training')
    assert(len(dataset.core) == 60000)
    image, label = dataset.get_example()
    assert(len(image.shape) == 3)
    assert(image.shape[0] == 28)
    assert(image.shape[1] == 28)
    assert(image.shape[2] == 1)
    assert(len(label.shape) == 0)
    assert(image.dtype == tf.float32)
    assert(label.dtype == tf.int64)

"""
if __name__ == "__main__":
    dataset = MNISTDataset(purpose='training')
    print()
    image, label = dataset.get_example()
    plt.figure()
    plt.imshow(image)
    plt.title(f"label={label.numpy()}")
    plt.show()
"""
