"""Base class for all datasets.
"""
import tensorflow as tf


class Dataset(object):

    PURPOSE_OPTIONS = ['training', 'evaluation']

    def __init__(self):
        self.core = None

    def shuffle(self, buffer_size=None, seed=None):
        """
        Returns:
            None.
        """
        if buffer_size is None:
            buffer_size = len(self.core)
        if seed is not None:
            tf.random.set_seed(seed)
        self.core = self.core.shuffle(buffer_size=buffer_size, seed=seed)

    def get_example(self):
        example = next(iter(self))
        return example
