from resize import Resize
import data
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = data.datasets.MNIST.MNISTDataset(purpose='training')
    example = dataset.example
    image, label = example['image'], example['label']
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.ToTensor(),
        Resize(size=(224, 224)),
    ])
    new_image, new_label = preprocessor(image, label)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title("Original image")
    ax2.imshow(new_image)
    ax2.set_title("Transformed image")
    plt.show()
