import tensorflow as tf
import tensorflow_datasets as tfds
from data.datasets import Dataset


class PASCALVOCDataset(Dataset):

    TASK_OPTIONS = ['image_classification', 'object_detection']
    NUM_CLASSES = 20
    MAPPING = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "diningtable",
        11: "dog",
        12: "horse",
        13: "motorbike",
        14: "person",
        15: "pottedplant",
        16: "sheep",
        17: "sofa",
        18: "train",
        19: "tvmonitor",
    }

    def __init__(self, purpose, task):
        super(PASCALVOCDataset, self).__init__()
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(f"[ERROR] Argument 'purpose' should be in {self.PURPOSE_OPTIONS}. Got {purpose}.")
        if task not in self.TASK_OPTIONS:
            raise ValueError(f"[ERROR] Argument 'task' should be one of {self.TASK_OPTIONS}. Got {task}.")
        self.task = task
        self.whole = tfds.load('voc', with_info=False)
        self.core = self.whole['train'] if purpose == 'training' else self.whole['validation']
        self.core = self.core.map(self.load_element)

    def load_element(self, element):
        image = tf.cast(element['image'], dtype=tf.float32)
        if self.task == 'image_classification':
            label = tf.cast(element['label'], dtype=tf.int64)
        elif self.task == 'object_detection':
            label = (tf.cast(element['objects']['bbox'], dtype=tf.float32),
                     tf.cast(element['objects']['label'], dtype=tf.int64))
        else:
            raise ValueError(f"[ERROR] Task {self.task} not recognized.")
        return image, label
