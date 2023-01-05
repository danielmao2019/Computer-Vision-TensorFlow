import tensorflow as tf


class SegNet(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(SegNet, self).__init__(**kwargs)
        self.num_classes = num_classes

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

    def _max_unpool_from_argmax(self, x, argmax):
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

    def _encoder_block(self, x, filters_list):
        for filters in filters_list:
            x = self._conv_layer(x, filters)
        x, argmax = self._max_pool_with_argmax(x)
        return x, argmax

    def _decoder_block(self, x, argmax, filters_list):
        x = self._max_unpool_from_argmax(x, argmax)
        for filters in filters_list:
            x = self._conv_layer(x, filters)
        return x

    def _encoder_network(self, x):
        """
        Args:
            x: The input feature map to the encoder network.
        Returns:
            x: The encoded feature map.
            argmax_list (list): A list of argmax info for the decoder network.
        """
        input_h, input_w = x.shape[1], x.shape[2]
        x, argmax_0 = self._encoder_block(x, [64, 64])
        assert x.shape[1] == input_h // 2 and x.shape[2] == input_w // 2, f"{x.shape=}, {input_h=}, {input_w=}"
        x, argmax_1 = self._encoder_block(x, [128, 128])
        assert x.shape[1] == input_h // 4 and x.shape[2] == input_w // 4, f"{x.shape=}, {input_h=}, {input_w=}"
        x, argmax_2 = self._encoder_block(x, [256, 256, 256])
        assert x.shape[1] == input_h // 8 and x.shape[2] == input_w // 8, f"{x.shape=}, {input_h=}, {input_w=}"
        x, argmax_3 = self._encoder_block(x, [512, 512, 512])
        assert x.shape[1] == input_h // 16 and x.shape[2] == input_w // 16, f"{x.shape=}, {input_h=}, {input_w=}"
        x, argmax_4 = self._encoder_block(x, [512, 512, 512])
        assert x.shape[1] == input_h // 32 and x.shape[2] == input_w // 32, f"{x.shape=}, {input_h=}, {input_w=}"
        argmax_list = [argmax_0, argmax_1, argmax_2, argmax_3, argmax_4]
        return x, argmax_list

    def _decoder_network(self, x, argmax_list):
        """
        Args:
            x: The input feature map to the decoder network.
            argmax_list (list): The list of argmax info from the encoder network.
        Returns:
            x: The decoded feature map.
        """
        # not sure about the number of filters. worked though
        input_h, input_w = x.shape[1], x.shape[2]
        x = self._decoder_block(x, argmax_list[4], [512, 512, 512])
        assert x.shape[1] == input_h * 2 and x.shape[2] == input_w * 2, f"{x.shape=}, {input_h=}, {input_w=}"
        x = self._decoder_block(x, argmax_list[3], [512, 512, 256])
        assert x.shape[1] == input_h * 4 and x.shape[2] == input_w * 4, f"{x.shape=}, {input_h=}, {input_w=}"
        x = self._decoder_block(x, argmax_list[2], [256, 256, 128])
        assert x.shape[1] == input_h * 8 and x.shape[2] == input_w * 8, f"{x.shape=}, {input_h=}, {input_w=}"
        x = self._decoder_block(x, argmax_list[1], [128, 64])
        assert x.shape[1] == input_h * 16 and x.shape[2] == input_w * 16, f"{x.shape=}, {input_h=}, {input_w=}"
        x = self._decoder_block(x, argmax_list[0], [64, self.num_classes])
        assert x.shape[1] == input_h * 32 and x.shape[2] == input_w * 32, f"{x.shape=}, {input_h=}, {input_w=}"
        return x

    def _classifier(self, x):
        """
        Returns:
            x: A 4-D tensor of shape (N, H, W, num_classes) representing the class probability map.
        """
        x = tf.keras.activations.softmax(x, axis=-1)
        return x

    def call(self, x):
        return self._classifier(self._decoder_network(*self._encoder_network(x)))

    def build(self, input_shape):
        assert type(input_shape) == tuple, f"{type(input_shape)=}"
        assert len(input_shape) == 3, f"{len(input_shape)=}"
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="SegNet",
        )
