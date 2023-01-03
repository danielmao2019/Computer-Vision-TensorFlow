import numpy as np
import tensorflow as tf
import data


class ToTensor(object):
    """This transform should be applied before any other component transforms is applied.
    """

    def __call__(self, image, label):
        """
        Arguments:
            image: A single image of type PIL image, numpy ndarray, tensorflow tensor, etc.
            label: The label of the image.
        """
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        if not isinstance(label, tf.Tensor):
            label = tf.convert_to_tensor(label, dtype=tf.int64)
        data.preprocess.image.sanity_check(image, label)
        return image, label
