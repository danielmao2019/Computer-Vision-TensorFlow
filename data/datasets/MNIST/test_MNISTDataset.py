from MNISTDataset import MNISTDataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = MNISTDataset(purpose='training')
    image, label = dataset.get_example()
    plt.figure()
    plt.imshow(image)
    plt.title(f"{label=}")
    plt.show()
