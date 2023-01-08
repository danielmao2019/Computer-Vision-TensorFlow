import tensorflow as tf


class MaxUnpoolFromArgmax(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxUnpoolFromArgmax, self).__init__(**kwargs)

    def call(self, x, argmax):
        """
        Args:
            x: A 4-D tensor of shape (N, H, W, C) and dtype tf.float32.
            argmax: A 4-D tensor of shape (N, H, W, C) and dtype tf.int64.
        Returns:
            output: A 4-D tensor of shape (N, 2*H, 2*W, C) and dtype tf.float32.
        """
        assert x.shape == argmax.shape, f"{x.shape=}, {argmax.shape=}"
        assert x.dtype == tf.float32, f"{x.dtype=}"
        assert argmax.dtype == tf.int64, f"{argmax.dtype=}"
        indices = tf.expand_dims(tf.reshape(argmax, shape=(-1,)), axis=-1)
        updates = tf.reshape(x, shape=(-1,))
        shape = tf.cast(4 * tf.math.reduce_prod(tf.shape(x), keepdims=True), dtype=tf.int64)
        output_flat = tf.scatter_nd(indices=indices, updates=updates, shape=shape)
        output = tf.reshape(output_flat, shape=(-1, x.shape[1]*2, x.shape[2]*2, x.shape[3]))
        assert output.dtype == tf.float32, f"{x.dtype=}"
        return output
