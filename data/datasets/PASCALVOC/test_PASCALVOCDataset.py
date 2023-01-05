import pytest
from data.datasets.PASCALVOC.PASCALVOCDataset import PASCALVOCDataset
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
