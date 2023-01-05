import tensorflow as tf
import tensorflow_datasets as tfds
from data.datasets import Dataset


class OxfordIIITPetsDataset(Dataset):

    NUM_CLASSES = 9
    TASK_OPTIONS = ['cls', 'seg']
    MAPPING = {
        0: 'cat',
        1: 'dog',
        2: 'bird',
        3: 'rodent',
        4: 'fish',
        5: 'reptile',
        6: 'amphibian',
        7: 'rabbit',
        8: 'hamster',
    }

    def __init__(self, purpose, task):
        super(OxfordIIITPetsDataset, self).__init__()
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(f"[ERROR] Argument \'purpose\' should be one of {self.PURPOSE_OPTIONS}. Got {purpose}.")
        if task not in self.TASK_OPTIONS:
            raise ValueError(f"[ERROR] Argument \'task\' should be one of {self.TASK_OPTIONS}. Got {task}.")
        self.task = task
        whole = tfds.load('oxford_iiit_pet:3.*.*', with_info=False)
        print(whole.keys())
        self.core = whole['train'] if purpose == 'training' else whole['test']
        self.core = self.core.map(self.load_element)

    def load_element(self, element):
        image = tf.cast(element['image'], dtype=tf.float32)
        if self.task == 'cls':
            label = tf.cast(element['label'], dtype=tf.int64)
        else:
            label = tf.cast(tf.squeeze(element['segmentation_mask']), dtype=tf.int64)
        # TODO: reenable these sanity checks.
        # assert len(image.shape) == 3, f"{image.shape=}"
        # assert len(label.shape) == 2, f"{label.shape=}"
        # assert image.shape[:2] == label.shape, f"{image.shape=}, {label.shape=}"
        return image, label
