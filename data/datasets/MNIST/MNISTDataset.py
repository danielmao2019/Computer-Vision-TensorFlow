import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MNISTDataset(object):

    NUM_CLASSES = 10

    def __init__(self, purpose):
        if purpose not in ['training', 'evaluation']:
            raise ValueError(f"[ERROR] Argument 'purpose' should be in ['training', 'evaluation']. Got {purpose}.")
        self.purpose = purpose
        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
        self.images = x_train if purpose == 'training' else x_eval
        self.labels = y_train if purpose == 'training' else y_eval
        self.images = np.expand_dims(self.images.astype(np.float32), axis=-1)
        self.labels = self.labels.astype(np.int64)
        assert len(self.images) == len(self.labels), f"{len(self.images)=}, {len(self.labels)=}"
        self.dataset_size = len(self.images)
        self.index = 0
        # select one example.
        idx = np.random.choice(range(self.dataset_size))
        self.example = {'image': self.images[idx], 'label': self.labels[idx]}

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.dataset_size:
            raise StopIteration
        image = self.images[self.index]
        label = self.labels[self.index]
        self.index += 1
        return image, label

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __str__(self):
        return (f"Dataset info:\n"
                f"size=self.dataset_size\n"
                f"image_shape={self.example['image'].shape}\n"
                f"label_shape={self.example['label'].shape}\n"
                f"num_classes={self.NUM_CLASSES}\n"
                )

    def tensorflow(self):
        def generator():
            return iter(self)
        image_spec = tf.TensorSpec(shape=self.example['image'].shape, dtype=tf.float32)
        label_spec = tf.TensorSpec(shape=self.example['label'].shape, dtype=tf.int64)
        return tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(image_spec, label_spec),
        )


if __name__ == "__main__":
    dataset = MNISTDataset(purpose='training')
    image, label = dataset.example['image'], dataset.example['label']
    print(f"{image.shape=}")
    print(f"{label.shape=}")
    plt.figure()
    plt.imshow(image)
    plt.title(f"label={label}")
    plt.show()
