import random


class Random(object):

    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, input, label):
        if random.random() < self.probability:
            input, label = self.transform(input, label)
        return input, label
