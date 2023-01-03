import tensorflow as tf


class ENet(tf.keras.Model):

    # TODO
    # implement self._max_pool_with_argmax and self._max_unpool_from_argmax as layers
    # to simplify model.summary().

    def __init__(self, num_classes, **kwargs):
        super(ENet, self).__init__(name="ENet", **kwargs)
        self._num_classes = num_classes

    def _initial_block(self, x):
        # activations???
        input_shape = x.shape
        assert input_shape[3] == 3
        conv = tf.keras.layers.Conv2D(
            filters=13, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
            name='initial_conv',
        )
        pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='initial_pool',
        )
        x = tf.keras.layers.Concatenate(
            name='initial_concat',
        )([conv(x), pool(x)])
        output_shape = x.shape
        assert output_shape[1] == input_shape[1] // 2
        assert output_shape[2] == input_shape[2] // 2
        assert output_shape[3] == 16
        return x

    def _projection_layer(self, x, filters, layer_id):
        projection = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="SAME", use_bias=False,
            name=layer_id + '_1x1',
        )
        batch_norm = tf.keras.layers.BatchNormalization(name=layer_id + '_batch_norm')
        prelu = tf.keras.layers.PReLU(name=layer_id + '_prelu')
        return prelu(batch_norm(projection(x)))

    def _regular_conv_layer(self, x, filters, kernel_size, strides, use_bias, layer_id):
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="SAME", use_bias=use_bias,
            name=layer_id + '_regular_conv',
        )
        batch_norm = tf.keras.layers.BatchNormalization(name=layer_id + '_batch_norm')
        prelu = tf.keras.layers.PReLU(name=layer_id + '_prelu')
        return prelu(batch_norm(conv(x)))

    def _dilated_conv_layer(self, x, filters, dilation_rate, layer_id):
        dilated_conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            dilation_rate=dilation_rate,
            name=layer_id + '_dilated_conv',
        )
        batch_norm = tf.keras.layers.BatchNormalization(name=layer_id + '_batch_norm')
        prelu = tf.keras.layers.PReLU(name=layer_id + '_prelu')
        return prelu(batch_norm(dilated_conv(x)))

    def _asymmetric_conv_layer(self, x, filters, layer_id):
        asymmetric_conv_1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(5, 1), strides=(1, 1), padding="SAME",
            name=layer_id + '_asymmetric_conv_v',
        )
        asymmetric_conv_2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 5), strides=(1, 1), padding="SAME",
            name=layer_id + '_asymmetric_conv_h',
        )
        batch_norm = tf.keras.layers.BatchNormalization(name=layer_id + '_batch_norm')
        prelu = tf.keras.layers.PReLU(name=layer_id + '_prelu')
        return prelu(batch_norm(asymmetric_conv_2(asymmetric_conv_1(x))))

    def _deconv_layer(self, x, filters, layer_id):
        deconv = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
            name=layer_id + '_deconv',
        )
        batch_norm = tf.keras.layers.BatchNormalization(name=layer_id + '_batch_norm')
        prelu = tf.keras.layers.PReLU(name=layer_id + '_prelu')
        return prelu(batch_norm(deconv(x)))

    def _max_pool_with_argmax(self, x, bottleneck_id):
        """
        Returns:
            (x, argmax) pair.
        """
        return tf.nn.max_pool_with_argmax(
            input=x, ksize=(2, 2), strides=(2, 2), padding="SAME",
        )

    def _max_unpool_from_argmax(self, x, argmax, bottleneck_id):
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

    def _conv_bottleneck(self, x, filters, spatial_dropout_rate, bottleneck_id, reduction=4,
                         dilation_rate=None, asymmetric=False):
        input_channels = x.shape[3]
        # compute main branch
        main = x
        # compute extension branch
        extension = x
        extension = self._projection_layer(
            extension, filters=input_channels//reduction,
            layer_id=bottleneck_id + '_reduction',
        )
        if dilation_rate is not None:
            assert asymmetric == False
            extension = self._dilated_conv_layer(
                extension, filters=input_channels//reduction, dilation_rate=dilation_rate,
                layer_id=bottleneck_id + '_conv',
            )
        elif asymmetric:
            extension = self._asymmetric_conv_layer(
                extension, filters=input_channels//reduction,
                layer_id=bottleneck_id + '_conv',
            )
        else:
            extension = self._regular_conv_layer(
                extension, filters=input_channels//reduction, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
                layer_id=bottleneck_id + '_conv',
            )
        extension = self._projection_layer(
            extension, filters=filters,
            layer_id=bottleneck_id + '_expansion',
        )
        extension = tf.keras.layers.SpatialDropout2D(
            rate=spatial_dropout_rate,
            name=bottleneck_id + '_spatial_dropout',
        )(extension)
        # combine two branches
        x = tf.keras.layers.PReLU(
            name=bottleneck_id + '_final_prelu',
        )(main + extension)
        return x

    def _downsampling_bottleneck(self, x, filters, spatial_dropout_rate, bottleneck_id, reduction=4):
        # why is it not conv2 that has stride 2?
        input_channels = x.shape[3]
        # compute main branch
        main = x
        main, argmax = self._max_pool_with_argmax(main, bottleneck_id=bottleneck_id)
        main = tf.pad(main, paddings=[[0, 0], [0, 0], [0, 0], [0, filters-input_channels]])
        # compute extension branch
        extension = x
        extension = self._regular_conv_layer(
            extension, filters=input_channels//reduction, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
            layer_id=bottleneck_id + '_reduction',
        )
        extension = self._regular_conv_layer(
            extension, filters=input_channels//reduction, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
            layer_id=bottleneck_id + '_conv',
        )
        extension = self._projection_layer(
            extension, filters=filters,
            layer_id=bottleneck_id + '_expansion',
        )
        extension = tf.keras.layers.SpatialDropout2D(
            rate=spatial_dropout_rate,
            name=bottleneck_id + '_spatial_dropout',
        )(extension)
        # combine two branches
        x = tf.keras.layers.PReLU(
            name=bottleneck_id + '_final_prelu',
        )(main + extension)
        return x, argmax

    def _upsampling_bottleneck(self, x, argmax, filters, spatial_dropout_rate, bottleneck_id, reduction=4):
        input_channels = x.shape[3]
        # compute main branch
        main = x
        main = tf.keras.layers.Conv2D(
            filters=argmax.shape[3], kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name=bottleneck_id + '_reduction',
        )(main)
        main = self._max_unpool_from_argmax(main, argmax, bottleneck_id=bottleneck_id)
        # compute extension branch
        extension = x
        extension = self._projection_layer(
            extension, filters=input_channels//reduction,
            layer_id=bottleneck_id + '_reduction',
        )
        extension = self._deconv_layer(
            extension, filters=input_channels//reduction,
            layer_id=bottleneck_id + '_conv',
        )
        extension = self._projection_layer(
            extension, filters=filters,
            layer_id=bottleneck_id + '_expansion',
        )
        extension = tf.keras.layers.SpatialDropout2D(
            rate=spatial_dropout_rate,
            name=bottleneck_id + '_spatial_dropout',
        )(extension)
        # combine two branches
        x = tf.keras.layers.PReLU(
            name=bottleneck_id + '_final_prelu',
        )(main + extension)
        return x

    def call(self, x):
        x = self._initial_block(x)
        # stage 1
        x, argmax1 = self._downsampling_bottleneck(x, filters=64, spatial_dropout_rate=0.01, bottleneck_id='1.0')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.01, bottleneck_id='1.1')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.01, bottleneck_id='1.2')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.01, bottleneck_id='1.3')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.01, bottleneck_id='1.4')
        # stage 2
        x, argmax2 = self._downsampling_bottleneck(x, filters=128, spatial_dropout_rate=0.1, bottleneck_id='2.0')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, bottleneck_id='2.1')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=2, bottleneck_id='2.2')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, asymmetric=True, bottleneck_id='2.3')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=4, bottleneck_id='2.4')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, bottleneck_id='2.5')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=8, bottleneck_id='2.6')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, asymmetric=True, bottleneck_id='2.7')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=16, bottleneck_id='2.8')
        # stage 3
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, bottleneck_id='3.0')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=2, bottleneck_id='3.1')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, asymmetric=True, bottleneck_id='3.2')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=4, bottleneck_id='3.3')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, bottleneck_id='3.4')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=8, bottleneck_id='3.5')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, asymmetric=True, bottleneck_id='3.6')
        x = self._conv_bottleneck(x, filters=128, spatial_dropout_rate=0.1, dilation_rate=16, bottleneck_id='3.7')
        # stage 4
        x = self._upsampling_bottleneck(x, argmax2, filters=64, spatial_dropout_rate=0.1, bottleneck_id='4.0')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.1, bottleneck_id='4.1')
        x = self._conv_bottleneck(x, filters=64, spatial_dropout_rate=0.1, bottleneck_id='4.2')
        # stage 5
        x = self._upsampling_bottleneck(x, argmax1, filters=16, spatial_dropout_rate=0.1, bottleneck_id='5.0')
        x = self._conv_bottleneck(x, filters=16, spatial_dropout_rate=0.1, bottleneck_id='5.1')
        # final deconv and softmax
        x = tf.keras.layers.Conv2DTranspose(
            filters=self._num_classes, kernel_size=(2, 2), strides=(2, 2), padding="SAME",
            name='fullconv',
        )(x)
        x = tf.keras.layers.Softmax(
            name='final_softmax',
        )(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="ENet",
        )


if __name__ == "__main__":
    model = ENet(num_classes=10)
    model = model.build(input_shape=(512, 512, 3))
    model.summary()
