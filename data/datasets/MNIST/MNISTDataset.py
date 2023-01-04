import tensorflow as tf
import tensorflow_datasets as tfds
from data.datasets import Dataset


class MNISTDataset(Dataset):

    NUM_CLASSES = 10

    def __init__(self, purpose):
        super(MNISTDataset, self).__init__()
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(f"[ERROR] Argument \'purpose\' should be in {self.PURPOSE_OPTIONS}. Got {purpose}.")
        whole = tfds.load('mnist', with_info=False)
        self.core = whole['train'] if purpose == 'training' else whole['test']

    def __len__(self):
        return len(self.core)

    def __iter__(self):
        for element in self.core:
            image = tf.cast(element['image'], dtype=tf.float32)
            label = tf.cast(element['label'], dtype=tf.int64)
            yield image, label

    def __str__(self):
        image, label = self.get_example()
        return (f"Dataset info:\n"
                f"size={len(self.core)}\n"
                f"image_shape={image.shape}, image_dtype={image.dtype}\n"
                f"label_shape={label.shape}, label_dtype={label.dtype}\n"
                f"num_classes={self.NUM_CLASSES}\n"
                )
