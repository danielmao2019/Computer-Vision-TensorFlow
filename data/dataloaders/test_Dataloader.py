from Dataloader import Dataloader
import data
from tqdm import tqdm


def test_map_and_batch(dataset, preprocessor, batch_size):
    dataloader = Dataloader(dataset=dataset, shuffle=True, preprocessor=preprocessor, batch_size=batch_size)
    print(f"{len(dataloader)=}")
    batch_image, batch_label = next(iter(dataloader))
    print(f"{batch_image.shape=}")
    print(f"{batch_label.shape=}")


def test_iteration(dataset):
    dataloader = Dataloader(dataset=dataset, shuffle=True)
    print(f"{len(dataloader)=}")
    class_count = {}
    for image, label in tqdm(dataloader):
        assert image.shape == (image_h, image_w, 1)
        class_count[label.numpy()] = class_count.get(label.numpy(), 0) + 1
    print(f"{class_count=}")


if __name__ == "__main__":
    image_h = image_w = 224
    dataset = data.datasets.MNISTDataset(purpose='training')
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.Resize(size=(image_h, image_w)),
    ])
    test_map_and_batch(dataset, preprocessor, batch_size=8)
    test_iteration(dataset)
