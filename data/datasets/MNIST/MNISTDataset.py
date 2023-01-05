import tensorflow as tf
import tensorflow_datasets as tfds
from data.datasets import Dataset


class MNISTDataset(Dataset):

    NUM_CLASSES = 10

    def __init__(self, purpose):
        super(MNISTDataset, self).__init__()
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(f"[ERROR] Argument \'purpose\' should be one of {self.PURPOSE_OPTIONS}. Got {purpose}.")
        whole = tfds.load('mnist', with_info=False)
        self.core = whole['train'] if purpose == 'training' else whole['test']
        self.core = self.core.map(self.load_element)

    def load_element(self, element):
        image = tf.cast(element['image'], dtype=tf.float32)
        label = tf.cast(element['label'], dtype=tf.int64)
        assert len(image.shape) == 3, f"{image.shape=}"
        assert len(label.shape) == 0, f"{label.shape=}"
        return image, label

    def __str__(self):
        image, label = self.get_example()
        return (f"Dataset info:\n"
                f"size={len(self.core)}\n"
                f"image_shape={image.shape}, image_dtype={image.dtype}\n"
                f"label_shape={label.shape}, label_dtype={label.dtype}\n"
                f"num_classes={self.NUM_CLASSES}\n"
                )
