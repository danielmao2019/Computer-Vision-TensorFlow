import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels, out_channels,
                 strides=(1, 1),
                 project_shortcut=False,
                 **kwargs):
        """__init__ method
        
        Arguments:
            in_channels (int): the number of channels of the inputs to the 2nd convolutional layer.
            out_channels (int): the number of channels of the outputs of the 3rd convolutional layer.
            strides (int, int): strides of the 2nd convolutional layer.
            project_shortcut (bool): use projection on shortcut.
        """
        super(ResidualBlock, self).__init__(name="ResidualBlock", **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._strides = strides
        self._project_shortcut = project_shortcut

        self.conv1 = tf.keras.layers.Conv2D(
            filters=self._in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self._in_channels,
            kernel_size=(3, 3),
            strides=self._strides,
            padding='same',
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self._out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
        )
        self.conv_shortcut = tf.keras.layers.Conv2D(
            filters=self._out_channels,
            kernel_size=(1, 1),
            strides=self._strides,
            padding='same',
        )

    def call(self, x):
        shortcut = x

        x = self.conv1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if self._project_shortcut or self._strides != (1, 1):
            shortcut = self.conv_shortcut(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.add([shortcut, x])
        x = tf.keras.layers.LeakyReLU()(x)

        return x
