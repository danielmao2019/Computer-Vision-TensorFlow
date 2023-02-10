import pytest
from data.datasets.PASCALVOC.PASCALVOCDataset import PASCALVOCDataset
import tensorflow_datasets as tfds
import tensorflow as tf

@pytest.mark.dependency()
def test_raw_dataset():
    whole = tfds.load('voc', with_info=False)
    assert set(whole.keys()) == set(['train', 'validation', 'test']), f"{set(whole.keys())=}"
    for task in ['train', 'validation', 'test']:
        dataset = whole[task]
        dataset = dataset.shuffle(buffer_size=len(dataset))
        element = next(iter(dataset))
        assert type(element) == dict, f"{type(element)=}"
        assert set(element.keys()) == set(['image', 'image/filename', 'labels', 'labels_no_difficult', 'objects']), f"{set(element.keys())=}"
        objects = element['objects']
        assert type(objects) == dict, f"{type(objects)=}"
        assert set(objects.keys()) == set(['bbox', 'is_difficult', 'is_truncated', 'label', 'pose']), f"{set(objects.keys())=}"

def test_PASCALVOC_dataset():
    dataset = PASCALVOCDataset(purpose='training', task='object_detection')
    assert(len(dataset) == 2501), f"{len(dataset)=}"
    example = dataset.get_example()
    assert(type(example) == tuple), f"{type(example)=}"
    assert(len(example) == 2), f"{len(example)=}"

    assert(len(example[0].get_shape()) == 3), f"{len(example[0].get_shape())=}"
    assert(example[0].get_shape() == (480, 389, 3)), f"{example[0].get_shape()=}"
    assert(example[0].dtype == tf.float32), f"{example[0].dtype=}"

    assert(type(example[1]) == tuple), f"{type(example[1])=}"
    assert(len(example[1]) == 2), f"{len(example[1])=}"
    assert(len(example[1][0].get_shape()) == 2), f"{len(example[1][0].get_shape())=}"
    assert(example[1][0].get_shape() == (4, 4)), f"{example[1][0].get_shape()=}"
    assert(example[1][0].dtype == tf.float32), f"{example[1][0].dtype=}"
    assert(len(example[1][1].get_shape()) == 1), f"{len(example[1][1].get_shape())=}"
    assert(example[1][1].get_shape() == (4)), f"{example[1][1].get_shape()=}"
    assert(example[1][1].dtype == tf.int64), f"{example[1][1].dtype=}"
