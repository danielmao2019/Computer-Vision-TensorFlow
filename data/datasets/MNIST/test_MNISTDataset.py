from MNISTDataset import MNISTDataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = MNISTDataset(purpose='training').tensorflow()
    dataset = dataset.batch(8)
    image_batch, label_batch = next(iter(dataset))
    image, label = image_batch[0], label_batch[0]
    plt.figure()
    plt.imshow(image)
    plt.title(f"Example image of digit {label}")
    plt.show()
