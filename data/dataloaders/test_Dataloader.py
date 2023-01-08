import pytest
import data
from tqdm import tqdm


# TODO: previous runs gave DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.
@pytest.mark.parametrize("purpose, batch_size, expected_length", [
    (  # test case
        'training', 1, 60000,
    ),
    (  # test case
        'training', 4, 15000,
    ),
    (  # test case
        'evaluation', 1, 10000,
    ),
    (  # test case
        'evaluation', 4, 2500,
    ),
])
def test_MNIST_batch_and_len(purpose, batch_size, expected_length):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    dataloader = data.dataloaders.Dataloader(dataset=dataset, shuffle=True, preprocessor=None, batch_size=batch_size)
    assert dataloader.batch_size == batch_size
    assert len(dataloader) == expected_length


@pytest.mark.parametrize("purpose, image_size", [
    (  # test case
        'training', (224, 224),
    ),
    (  # test case
        'evaluation', (128, 128),
    ),
])
def test_MNIST_map(purpose, image_size):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.Resize(size=image_size),
    ])
    dataloader = data.dataloaders.Dataloader(dataset=dataset, shuffle=True, preprocessor=preprocessor, batch_size=1)
    batch_image, batch_label = next(iter(dataloader))
    assert batch_image.shape == (1,) + image_size + (1,)
    assert batch_label.shape == (1,)


@pytest.mark.parametrize("purpose", [
    'training',
    'evaluation',
])
def test_MNIST_iter(purpose):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    dataloader = data.dataloaders.Dataloader(dataset=dataset, shuffle=True, preprocessor=None, batch_size=1)
    class_count = [0] * 10
    for _, label in tqdm(dataloader):
        class_count[int(label)] += 1
    if purpose == 'training':
        assert class_count == [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], f"{class_count=}"
    else:
        assert class_count == [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009], f"{class_count=}"
