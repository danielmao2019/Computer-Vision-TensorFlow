import tensorflow as tf


class FCN(tf.keras.Model):
    """
    Currently only support VGG16 as the base model.
    """

    def __init__(self, num_classes, **kwargs):
        super(FCN, self).__init__(name="FCN", **kwargs)
        self._num_classes = num_classes

    def _conv_layer(self, x, filters, layer_id):
        """
        conv + bn + relu.
        """
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), padding="SAME", activation=None,
            name='conv_' + layer_id,
        )
        batch_normalization = tf.keras.layers.BatchNormalization(
            name='batch_norm_' + layer_id,
        )
        relu = tf.keras.layers.ReLU(
            name='relu_' + layer_id,
        )
        return relu(batch_normalization(conv(x)))

    def _conv_block(self, x, filters_list, block_id):
        for idx, filters in enumerate(filters_list):
            x = self._conv_layer(x, filters, layer_id=block_id + str(idx+1))
        max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='max_pool_' + block_id,
        )
        x = max_pool(x)
        return x

    def _backbone_network(self, x):
        x = self._conv_block(x, [64, 64], block_id='a')
        x = self._conv_block(x, [128, 128], block_id='b')
        x = self._conv_block(x, [256, 256, 256], block_id='c')
        pool3 = x
        x = self._conv_block(x, [512, 512, 512], block_id='d')
        pool4 = x
        x = self._conv_block(x, [512, 512, 512], block_id='e')
        pool5 = x
        return pool3, pool4, pool5

    def call(self, x):
        pool3, pool4, pool5 = self._backbone_network(x)
        # compute fcn_32s
        pool5 = tf.keras.layers.Conv2D(
            filters=self._num_classes, kernel_size=(1, 1), strides=(1, 1),
            name='proj_pool5',
        )(pool5)
        fcn_32s = tf.keras.layers.UpSampling2D(
            size=(32, 32), name='upsample_32',
        )(pool5)
        # compute fcn_16s
        pool4 = tf.keras.layers.Conv2D(
            filters=self._num_classes, kernel_size=(1, 1), strides=(1, 1),
            name='proj_pool4',
        )(pool4)
        pool4 = pool4 + tf.keras.layers.UpSampling2D(
            size=(2, 2), name='upsample_pool5',
        )(pool5)
        fcn_16s = tf.keras.layers.UpSampling2D(
            size=(16, 16), name='upsample_16',
        )(pool4)
        # compute fcn_8s
        pool3 = tf.keras.layers.Conv2D(
            filters=self._num_classes, kernel_size=(1, 1), strides=(1, 1),
            name='proj_pool3',
        )(pool3)
        pool3 = pool3 + tf.keras.layers.UpSampling2D(
            size=(2, 2), name='upsample_pool4',
        )(pool4)
        fcn_8s = tf.keras.layers.UpSampling2D(
            size=(8, 8), name='upsample_8',
        )(pool3)
        return fcn_32s, fcn_16s, fcn_8s

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="FCN",
        )


if __name__ == "__main__":
    model = FCN(num_classes=21)
    model = model.build(input_shape=(512, 512, 3))
    model.summary()
