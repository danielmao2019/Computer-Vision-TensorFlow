import tensorflow as tf
import tensorflow_datasets as tfds
import logging; logging.getLogger().setLevel(logging.INFO)

from frameworks.detection.preprocessor import Preprocessor
from frameworks.detection.encoder import Encoder


class DataLoader:

    def __init__(self, dataset_name=r"coco/2017", dataset_dir=None,
                 image_height=512, image_width=512, batch_size=8,
                 encoder=None, preprocessor=None):
        self.DATASET_NAME = dataset_name
        self.DATASET_DIR = dataset_dir
        self.BATCH_SIZE = batch_size
        self.encoder = encoder if encoder else Encoder()
        self.preprocessor = preprocessor if preprocessor else Preprocessor(image_height, image_width)

    def get_train_valid_datasets(self, num_train=None, num_valid=None):
        (train_dataset, valid_dataset), dataset_info = tfds.load(
            name=self.DATASET_NAME, split=["train", "validation"], data_dir=self.DATASET_DIR,
            with_info=True,
        )
        if num_train:
            train_dataset = train_dataset.take(num_train)
        else:
            num_train = dataset_info.splits['train'].num_examples
        logging.info(f"Number of training examples: {num_train}")
        if num_valid:
            valid_dataset = valid_dataset.take(num_valid)
        else:
            num_valid = dataset_info.splits['validation'].num_examples
        logging.info(f"Number of validation examples: {num_valid}")
        autotune = tf.data.AUTOTUNE
        ### Create training dataset
        train_dataset = train_dataset.map(self.preprocessor.preprocess, num_parallel_calls=autotune)
        # train_dataset = train_dataset.shuffle(8 * self.BATCH_SIZE)
        train_dataset = train_dataset.padded_batch(batch_size=self.BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
        train_dataset = train_dataset.map(self.encoder.encode, num_parallel_calls=autotune)
        train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
        train_dataset = train_dataset.prefetch(autotune)
        ### Create validation dataset
        valid_dataset = valid_dataset.map(self.preprocessor.preprocess, num_parallel_calls=autotune)
        valid_dataset = valid_dataset.padded_batch(batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
        valid_dataset = valid_dataset.map(self.encoder.encode, num_parallel_calls=autotune)
        valid_dataset = valid_dataset.apply(tf.data.experimental.ignore_errors())
        valid_dataset = valid_dataset.prefetch(autotune)
        return train_dataset, valid_dataset, dataset_info
