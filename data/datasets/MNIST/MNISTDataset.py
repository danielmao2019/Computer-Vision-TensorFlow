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
        self.core = self.core.map(lambda example: (
            tf.cast(example['image'], dtype=tf.float32),
            tf.cast(example['label'], dtype=tf.int64),
        ))
        assert isinstance(self.core, tf.data.Dataset), f"{type(self.core)=}"

    def get_example(self):
        image, label = next(iter(self.core))
        assert isinstance(image, tf.Tensor), f"{type(image)=}"
        assert len(image.shape) == 3, f"{image.shape=}"
        assert image.dtype == tf.float32, f"{image.dtype=}"
        assert isinstance(label, tf.Tensor), f"{type(label)=}"
        assert len(label.shape) == 0, f"{label.shape=}"
        assert label.dtype == tf.int64, f"{label.dtype=}"
        return image, label

    def __str__(self):
        image, label = self.get_example()
        return (f"Dataset info:\n"
                f"size={len(self.core)}\n"
                f"image_shape={image.shape}, image_dtype={image.dtype}\n"
                f"label_shape={label.shape}, label_dtype={label.dtype}\n"
                f"num_classes={self.NUM_CLASSES}\n"
                )
