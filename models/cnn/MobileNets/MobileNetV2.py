import tensorflow as tf


class MobileNetV2(tf.keras.Model):
    """
    Reference: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, output_dim, **kwargs):
        super(MobileNetV2, self).__init__(name="MobileNetV2", **kwargs)
        self._output_dim = output_dim

    def _bottleneck_block_v2(self, x, ich, och, strides, expansion_factor):
        """
        Arguments:
            x (tensor): input to the bottleneck block.
            ich (int): number of input channels.
            och (int): number of output channels.
            strides (int): strides for the depthwise convolutional layer.
            expansion_factor (float): expansion factor from num input channels to num intermediate channels.
        Returns:
            Processed input (x).
        """
        shortcut = x
        intermediate_channels = ich * expansion_factor
        exp_conv = tf.keras.layers.Conv2D(
            filters=intermediate_channels, kernel_size=1, strides=1, padding="same",
        )
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=strides, padding="same",
        )
        out_conv = tf.keras.layers.Conv2D(
            filters=och, kernel_size=1, strides=1, padding="same",
        )
        relu6 = tf.keras.layers.ReLU(
            max_value=6, negative_slope=0, threshold=0,
        )
        x = exp_conv(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = relu6(x)
        x = depthwise_conv(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = relu6(x)
        x = out_conv(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # ??? one batch norm or 3?
        if x.shape == shortcut.shape:
            x = tf.keras.layers.Add()([x, shortcut])
        return x

    def _bottleneck_sequence(self, x, ich, och, s, t, n):
        """
        Arguments:
            x (tensor): input to the bottleneck sequence.
            ich (int): number of input channels.
            och (int): number of output channels.
            s (int): strides for the first bottleneck block in the sequence.
            t (float): expansion factor for each bottleneck block in the sequence.
            n (int): number of bottleneck blocks in the sequence.
        Returns:
            Processed input (x).
        """
        for i in range(n):
            strides = s if i == 0 else 1
            x = self._bottleneck_block_v2(x, ich=ich, och=och, strides=strides, expansion_factor=t)
        return x

    def call(self, x):
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self._bottleneck_sequence(x, ich=32, och=16, s=1, t=1, n=1)
        x = self._bottleneck_sequence(x, ich=16, och=24, s=2, t=6, n=2)
        x = self._bottleneck_sequence(x, ich=24, och=32, s=2, t=6, n=3)
        x = self._bottleneck_sequence(x, ich=32, och=64, s=2, t=6, n=4)
        x = self._bottleneck_sequence(x, ich=64, och=96, s=1, t=6, n=3)
        x = self._bottleneck_sequence(x, ich=96, och=160, s=2, t=6, n=3)
        x = self._bottleneck_sequence(x, ich=160, och=320, s=1, t=6, n=1)
        x = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=1, strides=1, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = tf.keras.layers.Conv2D(
            filters=self._output_dim, kernel_size=1, strides=1, padding="valid",
        )(x)
        x = tf.reshape(x, shape=(self._output_dim,))
        return x

    def build(self, input_shape=(224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
            name="MobileNetV2",
        )


if __name__ == "__main__":
    model = MobileNetV2(output_dim=10)
    model = model.build()
    model.summary()
    # tf.keras.utils.plot_model(model, to_file="MobileNetV2.png", show_shapes=True)
