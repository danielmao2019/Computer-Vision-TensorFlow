import tensorflow as tf


class SegNet(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(SegNet, self).__init__(name="SegNet", **kwargs)
        self._num_classes = num_classes

    def _conv_layer(self, x, filters):
        """
        conv + bn + relu.
        """
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), padding="SAME",
        )
        batch_normalization = tf.keras.layers.BatchNormalization()
        relu = tf.keras.activations.relu
        return relu(batch_normalization(conv(x)))

    def _max_pool_with_argmax(self, x):
        """
        Returns:
            (x, argmax) pair.
        """
        return tf.nn.max_pool_with_argmax(
            input=x, ksize=(2, 2), strides=(2, 2), padding="SAME",
        )

    def _max_unpool_from_argmax(self, x, argmax):
        if x.shape[1:] != argmax.shape[1:]:
            raise ValueError(f"[ERROR] x.shape {x.shape} and argmax.shape {argmax.shape} do not match.")
        return tf.reshape(
            tensor=tf.scatter_nd(
                indices=tf.expand_dims(tf.reshape(argmax, [-1]), axis=-1),
                updates=tf.reshape(x, [-1]),
                shape=[tf.math.reduce_prod(tf.shape(x))],
            ),
            shape=[-1, x.shape[1] * 2, x.shape[2] * 2, x.shape[3]],
        )

    def _encoder_block(self, x, filters_list):
        for filters in filters_list:
            x = self._conv_layer(x, filters)
        x, argmax = self._max_pool_with_argmax(x)
        return x, argmax

    def _decoder_block(self, x, argmax, filters_list):
        assert x.shape[1:] == argmax.shape[1:], f"{x.shape} != {argmax.shape}"
        x = self._max_unpool_from_argmax(x, argmax)
        for filters in filters_list:
            x = self._conv_layer(x, filters)
        return x

    def _encoder_network(self, x):
        """
        Returns:
            x: encoded feature map.
            argmax_list: list of argmax info.
        """
        x, argmax_0 = self._encoder_block(x, [64, 64])
        x, argmax_1 = self._encoder_block(x, [128, 128])
        x, argmax_2 = self._encoder_block(x, [256, 256, 256])
        x, argmax_3 = self._encoder_block(x, [512, 512, 512])
        x, argmax_4 = self._encoder_block(x, [512, 512, 512])
        argmax_list = [argmax_0, argmax_1, argmax_2, argmax_3, argmax_4]
        return x, argmax_list

    def _decoder_network(self, x, argmax_list):
        """
        Arguments:
            x: input feature map to the decoder network.
            argmax_list: list of argmax info from the encoder network.
        """
        # not sure about the number of filters. worked though
        x = self._decoder_block(x, argmax_list[4], [512, 512, 512])
        x = self._decoder_block(x, argmax_list[3], [512, 512, 256])
        x = self._decoder_block(x, argmax_list[2], [256, 256, 128])
        x = self._decoder_block(x, argmax_list[1], [128, 64])
        x = self._decoder_block(x, argmax_list[0], [64, self._num_classes])
        return x

    def _classifier(self, x):
        """
        Returns:
            probability_map: 3-D tensor of shape (H, W, K).
        """
        x = tf.keras.activations.softmax(x, axis=-1)
        return x

    def call(self, x):
        return self._classifier(self._decoder_network(*self._encoder_network(x)))

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="SegNet",
        )


if __name__ == "__main__":
    model = SegNet(num_classes=10)
    model = model.build(input_shape=(512, 512, 3))
    model.summary()
