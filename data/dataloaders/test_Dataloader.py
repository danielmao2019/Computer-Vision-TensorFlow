from Dataloader import Dataloader
import data
from tqdm import tqdm


if __name__ == "__main__":
    image_h = image_w = 224
    dataset = data.datasets.MNISTDataset(purpose='training')
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.Resize(size=(image_h, image_w)),
    ])
    dataloader = Dataloader(dataset=dataset, shuffle=True, preprocessor=preprocessor)
    class_count = {}
    for image, label in tqdm(dataloader):
        assert image.shape == (image_h, image_w, 1)
        class_count[label.numpy()] = class_count.get(label.numpy(), 0) + 1
    print(f"{class_count=}")
