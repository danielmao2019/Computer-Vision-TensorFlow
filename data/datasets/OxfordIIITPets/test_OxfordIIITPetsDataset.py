import matplotlib.pyplot as plt
import pytest
from data.datasets.OxfordIIITPets.OxfordIIITPetsDataset import OxfordIIITPetsDataset
import tensorflow_datasets as tfds
import tensorflow as tf

# unit tests checking example from MNIST dataset for training asserting data shape, type, and load count
def test_OxfordIIITPetsDataset_dataset():
    dataset = OxfordIIITPetsDataset(purpose='training', task='semantic_segmentation')
    assert(len(dataset.core) == 3680), f"{len(dataset.core)=}"
    example = dataset.get_example()
    assert(type(example) == tuple), f"{type(example)=}"
    assert(len(example) == 2), f"{len(example)=}"

    assert(len(example[0].get_shape()) == 3), f"{len(example[0].get_shape())=}"
    assert(example[0].get_shape() == (500, 500, 3)), f"{example[0].get_shape()=}"
    assert(example[0].dtype == tf.float32), f"{example[0].dtype=}"

    assert(len(example[1].get_shape()) == 2), f"{len(example[1].get_shape())=}"
    assert(example[1].get_shape() == (500, 500)), f"{example[1].get_shape()=}"
    assert(example[1].dtype == tf.int64), f"{example[1].dtype=}"
