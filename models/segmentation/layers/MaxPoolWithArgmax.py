import tensorflow as tf


class MaxPoolWithArgmax(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxPoolWithArgmax, self).__init__(**kwargs)

    def call(self, x):
        """
        Args:
            x: A 4-D tensor of shape (N, H, W, C) and dtype tf.float32.
        Returns:
            x: A 4-D tensor of shape (N, H/2, W/2, C) and dtype tf.float32.
            argmax: A 4-D tensor of shape (N, H/2, W/2, C) and dtype tf.int64.
        """
        assert x.dtype == tf.float32, f"{x.dtype=}"
        x, argmax = tf.nn.max_pool_with_argmax(
            input=x, ksize=(2, 2), strides=(2, 2), padding="SAME",
        )
        assert x.shape == argmax.shape, f"{x.shape=}, {argmax.shape=}"
        assert x.dtype == tf.float32, f"{x.dtype=}"
        assert argmax.dtype == tf.int64, f"{argmax.dtype=}"
        return x, argmax
