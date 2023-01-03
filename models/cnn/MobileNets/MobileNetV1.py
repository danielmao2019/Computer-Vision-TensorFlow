import tensorflow as tf


class MobileNetV1(tf.keras.Model):
    """
    Reference: https://arxiv.org/abs/1704.04861.
    """

    def __init__(self, width_multiplier=1, resolution_multiplier=1, **kwargs):
        """
        Arguments:
            width_multiplier (float): width multiplier of the model.
            resolution_multiplier (float): resolution multiplier of the model.
        """
        super(MobileNetV1, self).__init__(name="MobileNetV1", **kwargs)
        self._width_multiplier = width_multiplier
        self._resolution_multiplier = resolution_multiplier

    def _conv_block(self, x, filters, kernel_size, strides):
        filters = int(filters * self._width_multiplier)
        conv_layer = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
        )
        bn_layer = tf.keras.layers.BatchNormalization()
        x = conv_layer(x)
        x = bn_layer(x)
        x = tf.keras.activations.relu(x)
        return x

    def _separable_conv_block(self, x, filters, kernel_size, strides):
        """
        Arguments:
            x (tensor): input to the separable convolution block.
            filters (int): filters of the pointwise convolution.
            kernel_size (int): kernel size of the depthwise convolution.
            strides (int): strides of the depthwise convolution.
        """
        filters = int(filters * self._width_multiplier)
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding="same",
        )
        batch_norm_1 = tf.keras.layers.BatchNormalization()
        pointwise_conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=1, padding="same",
        )
        batch_norm_2 = tf.keras.layers.BatchNormalization()
        x = depthwise_conv(x)
        x = batch_norm_1(x)
        x = tf.keras.activations.relu(x)
        x = pointwise_conv(x)
        x = batch_norm_2(x)
        x = tf.keras.activations.relu(x)
        return x

    def call(self, x):
        x = tf.image.resize(x, size=(
            x.shape[-3] * self._resolution_multiplier,
            x.shape[-2] * self._resolution_multiplier))
        x = self._conv_block(x, filters=32, kernel_size=3, strides=2)
        x = self._separable_conv_block(x, filters=64, kernel_size=3, strides=1)
        x = self._separable_conv_block(x, filters=128, kernel_size=3, strides=2)
        x = self._separable_conv_block(x, filters=128, kernel_size=3, strides=1)
        x = self._separable_conv_block(x, filters=256, kernel_size=3, strides=2)
        x = self._separable_conv_block(x, filters=256, kernel_size=3, strides=1)
        x = self._separable_conv_block(x, filters=512, kernel_size=3, strides=2)
        for _ in range(5):
            x = self._separable_conv_block(x, filters=512, kernel_size=3, strides=1)
        for _ in range(2):
            x = self._separable_conv_block(x, filters=1024, kernel_size=3, strides=2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(units=1000)(x)
        x = tf.keras.activations.softmax(x)
        return x

    def build(self, input_shape=(224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
            name="MobileNetV1",
        )


if __name__ == "__main__":
    model = MobileNetV1()
    model = model.build()
    model.summary()
