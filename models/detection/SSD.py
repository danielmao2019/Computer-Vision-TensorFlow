import tensorflow as tf


class SSD(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(SSD, self).__init__(name="SSD", **kwargs)
        self._num_classes = num_classes

    def _conv_layer(self, x, filters, kernel_size, conv_strides=1,
                    padding='same', activation='linear',
                    pool=False, pool_size=2, pool_strides=2, conv=True):
        if conv == True:
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=conv_strides,
                activation=activation,
                padding=padding,
                kernel_initializer='he_normal',
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        elif pool == True:
            x = tf.keras.layers.MaxPool2D(
                pool_size=pool_size, strides=pool_strides, padding='same',
            )(x)
        return x

    def _get_output(self, x, filters, outputs):
        output = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='glorot_normal',
        )(x)
        output = tf.keras.layers.Reshape([-1, 4+self._num_classes+1])(output)
        outputs.append(output)

    def call(self, x):
        outputs = []
        densenet_121 = tf.keras.applications.DenseNet121(
            input_shape=(x.shape[1], x.shape[2], 3), include_top=False,
        )
        #Feature Layer 1
        layer = densenet_121.get_layer('pool3_relu').output
        self._get_output(
            x=layer, filters=4*(4+self._num_classes+1), outputs=outputs,
        )
        #Feature Layer 2
        layer = densenet_121.get_layer('pool4_relu').output
        self._get_output(
            x=layer, filters=6*(4+self._num_classes+1), outputs=outputs,
        )
        #Feature Layer 3
        layer = densenet_121.get_layer('relu').output
        self._get_output(
            x=layer, filters=6*(4+self._num_classes+1), outputs=outputs,
        )
        #Feature Layer 4
        layer = self._conv_layer(128, 1, layer)
        layer = self._conv_layer(256, 3, layer, conv_strides=2)
        self._get_output(
            x=layer, filters=6*(4+self._num_classes+1), outputs=outputs,
        )
        #Feature Layer 5
        layer = self._conv_layer(128, 1, layer, padding='valid')
        layer = self._conv_layer(256, 3, layer, padding='valid')
        self._get_output(
            x=layer, filters=4*(4+self._num_classes+1), outputs=outputs,
        )       
        #Feature Layer 6
        layer = self._conv_layer(128, 1, layer, padding='valid')
        layer = self._conv_layer(256, 3, layer, padding='valid')
        self._get_output(
            x=layer, filters=4*(4+self._num_classes+1), outputs=outputs,
        )

        return tf.keras.layers.Concatenate(axis=1)(outputs)


if __name__ == "__main__":
    model = SSD(num_classes=10)
    model.build(input_shape=(1, 224, 224, 3))
    model.summary()
