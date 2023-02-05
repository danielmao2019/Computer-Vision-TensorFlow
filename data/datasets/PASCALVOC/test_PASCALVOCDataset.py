import pytest
from data.datasets.PASCALVOC.PASCALVOCDataset import PASCALVOCDataset
# from PASCALVOCDataset import PASCALVOCDataset
import tensorflow_datasets as tfds


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
    assert(len(dataset) == 2501)
    example = dataset.get_example()
    assert(type(example) == tuple)
    assert(len(example) == 2)

    assert(len(example[0].get_shape()) == 3)
    assert(example[0].get_shape()[0] == 480)
    assert(example[0].get_shape()[1] == 389)
    assert(example[0].get_shape()[2] == 3)
    assert(example[0].dtype == 'float32')

    assert(type(example[1]) == tuple)
    assert(len(example[1]) == 2)
    assert(len(example[1][0].get_shape()) == 2)
    assert(example[1][0].get_shape()[0] == 4)
    assert(example[1][0].get_shape()[1] == 4)
    assert(example[1][0].dtype == 'float32')
    assert(len(example[1][1].get_shape()) == 1)
    assert(example[1][1].get_shape()[0] == 4)
    assert(example[1][1].dtype == 'int64')
