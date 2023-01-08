import tensorflow as tf
import data


class Resize(object):

    def __init__(self, size=None, scale=None):
        if size is None and scale is None:
            raise ValueError(f"[ERROR] Exactly one of the arguments 'size' and 'scale' should be provided. Got none of them.")
        if size is not None and scale is not None:
            raise ValueError(f"[ERROR] Exactly one of the arguments 'size' and 'scale' should be provided. Got both of them.")
        if size and isinstance(size, int):
            size = (size, size)
        self.size = size
        if scale and isinstance(scale, float):
            scale = (scale, scale)
        self.scale = scale

    def __call__(self, image, label):
        data.preprocess.image.sanity_check(image, label)
        if self.size:
            new_h, new_w = self.size
        if self.scale:
            new_h = image.shape[-3] * self.scale[0]
            new_w = image.shape[-2] * self.scale[1]
        if len(label.shape) == 0:
            pass
        elif len(label.shape) == 2 and label.shape == image.shape[:2]:
            label = tf.squeeze(tf.image.resize(tf.expand_dims(label, axis=-1), size=(new_h, new_w), method='nearest'))
        else:
            raise ValueError(f"[ERROR] Argument 'label' shape out of control. Got {image.shape=} and {label.shape=}.")
        image = tf.image.resize(image, size=(new_h, new_w))
        data.preprocess.image.sanity_check(image, label)
        return image, label
