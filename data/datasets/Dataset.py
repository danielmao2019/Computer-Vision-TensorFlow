"""Base class for all datasets.
"""
import tensorflow as tf


class Dataset(object):

    PURPOSE_OPTIONS = ['training', 'evaluation']

    def __init__(self):
        self.core = None

    def __len__(self):
        return len(self.core)

    def __iter__(self):
        for element in self.core:
            yield element

    def get_example(self):
        return next(iter(self.core))

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
        return self

    def map(self, func):
        self.core = self.core.map(func)
        return self

    def batch(self, batch_size):
        self.core = self.core.batch(batch_size)
        return self

    def unbatch(self):
        self.core = self.core.unbatch()
        return self

    def take(self, count):
        self.core = self.core.take(count)
        return self
