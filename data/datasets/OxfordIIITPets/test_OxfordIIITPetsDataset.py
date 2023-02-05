import matplotlib.pyplot as plt
import pytest
# from OxfordIIITPetsDataset import OxfordIIITPetsDataset # use this import if running from Computer-Vision-TensorFlow/data/datasets/OxfordIIITPets
from data.datasets.OxfordIIITPets.OxfordIIITPetsDataset import OxfordIIITPetsDataset # use this import if running pytest from Computer-Vision-TensorFlow
import tensorflow_datasets as tfds
import tensorflow as tf

# unit tests checking example from MNIST dataset for training asserting data shape, type, and load count
def test_OxfordIIITPetsDataset_dataset():
    dataset = OxfordIIITPetsDataset(purpose='training', task='semantic_segmentation')
    assert(len(dataset.core) == 3680)
    example = dataset.get_example()
    assert(type(example) == tuple)
    assert(len(example) == 2)

    assert(len(example[0].get_shape()) == 3)
    assert(example[0].get_shape()[0] == 500)
    assert(example[0].get_shape()[1] == 500)
    assert(example[0].get_shape()[2] == 3)
    assert(example[0].dtype == 'tf.float32')

    assert(len(example[1].get_shape()) == 2)
    assert(example[1].get_shape()[0] == 500)
    assert(example[1].get_shape()[1] == 500)
    assert(example[1].dtype == 'int64')


"""
if __name__ == "__main__":
    dataset = OxfordIIITPetsDataset(purpose='training', task='semantic_segmentation')
    print(len(dataset))
    example = dataset.get_example()
    print(example)
    print(example[0].dtype)
    print(type(example))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(tf.cast(example[0], dtype=tf.int64))
    ax1.set_title('Image')
    ax2.imshow(example[1])
    ax2.set_title('Segmentation Mask')
    plt.show()
"""
