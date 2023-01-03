import tensorflow as tf


class FPN(tf.keras.Model):
    """
    Currently using ResNet 18 as backbone only.
    """

    def __init__(self, feature_dimension, **kwargs):
        super(FPN, self).__init__(name="FPN", **kwargs)
        self._feature_dimension = feature_dimension

    def _stage_1_and_max_pool(self, x):
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME",
            name='stage_1',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding="SAME",
            name='stage_2_max_pool',
        )(x)
        return x

    def _conv_layer(self, x, filters, strides, layer_id):
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), strides=strides, padding="SAME",
            name=layer_id + '_conv',
        )
        batch_normalization = tf.keras.layers.BatchNormalization(
            name=layer_id + '_batch_norm',
        )
        relu = tf.keras.layers.ReLU(
            name=layer_id + '_relu',
        )
        return relu(batch_normalization(conv(x)))

    def _reduction_layer(self, x, filters, layer_id):
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name=layer_id + '_projection',
        )
        batch_normalization = tf.keras.layers.BatchNormalization(
            name=layer_id + '_batch_norm',
        )
        relu = tf.keras.layers.ReLU(
            name=layer_id + '_relu',
        )
        return relu(batch_normalization(conv(x)))

    def _expansion_layer(self, x, filters, layer_id):
        conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name=layer_id + '_projection',
        )
        batch_normalization = tf.keras.layers.BatchNormalization(
            name=layer_id + '_batch_norm',
        )
        return batch_normalization(conv(x))

    def _regular_block(self, x, filters, strides, block_id):
        """
        Arguments:
            filters (int): number of filters for both conv layers.
            strides (int): strides for the Conv2D in the first self._conv_layer.
        """
        shortcut = x
        x = self._conv_layer(x, filters=filters, strides=strides, layer_id=block_id + '_layer_1')
        x = self._conv_layer(x, filters=filters, strides=(1, 1), layer_id=block_id + '_layer_2')
        if x.shape[3] != shortcut.shape[3]:
            assert x.shape[3] == shortcut.shape[3] * 2
            assert shortcut.shape[1] == x.shape[1] * 2
            assert shortcut.shape[2] == x.shape[2] * 2
            shortcut = tf.keras.layers.Conv2D(
                filters=x.shape[3], kernel_size=(1, 1), strides=(2, 2), padding="SAME",
                name=block_id + '_projection',
            )(shortcut)
        x = tf.keras.layers.Add(
            name=block_id + '_merge',
        )([x, shortcut])
        x = tf.keras.layers.ReLU(
            name=block_id + '_final_relu',
        )(x)
        return x

    def _bottleneck_block(self, x, filters, block_id):
        """
        Arguments:
            filters (int): number of filters for reduction and conv. The number of filters for expansion is 4 * filters.
        """
        shortcut = x
        x = self._reduction_layer(x, filters=filters, layer_id=block_id + '_layer_1')
        x = self._conv_layer(x, filters=filters, layer_id=block_id + '_layer_2')
        x = self._expansion_layer(x, filters=filters * 4, layer_id=block_id + '_layer_3')
        if x.shape[3] != shortcut.shape[3]:
            shortcut = tf.keras.layers.Conv2D(
                filters=x.shape[3], kernel_size=(1, 1), strides=(1, 1), padding="SAME",
                name=block_id + '_projection',
            )(shortcut)
        x = tf.keras.layers.Add(
            name=block_id + '_merge',
        )([x, shortcut])
        x = tf.keras.layers.ReLU(
            name=block_id + '_final_relu',
        )(x)
        return x

    def _regular_stage(self, x, filters, strides, num_blocks, stage_id):
        for block_idx in range(num_blocks):
            strides = strides if block_idx == 0 else (1, 1)
            x = self._regular_block(x, filters=filters, strides=strides, block_id=stage_id + f"{block_idx+1}")
        return x

    def _bottleneck_stage(self, x, filters, num_blocks, stage_id):
        for block_idx in range(num_blocks):
            strides = (2, 2) if block_idx == 0 else (1, 1)
            x = self._bottleneck_block(x, filters=filters, block_id=stage_id + f"{block_idx+1}")
        return x

    def _bottom_up_pathway(self, x, block_type, num_blocks_list):
        if block_type == 'regular':
            func = self._regular_stage
        elif block_type == 'bottleneck':
            func = self._bottleneck_stage
        else:
            raise ValueError(f"[ERROR] block_type must be in ['regular', 'bottleneck']. Got {block_type}.")
        assert len(num_blocks_list) == 4
        x = self._stage_1_and_max_pool(x)
        x = func(x, filters=64, strides=(1, 1), num_blocks=num_blocks_list[0], stage_id="block_2.")
        C2 = x
        x = func(x, filters=128, strides=(2, 2), num_blocks=num_blocks_list[1], stage_id="block_3.")
        C3 = x
        x = func(x, filters=256, strides=(2, 2), num_blocks=num_blocks_list[2], stage_id="block_4.")
        C4 = x
        x = func(x, filters=512, strides=(2, 2), num_blocks=num_blocks_list[3], stage_id="block_5.")
        C5 = x
        return [C2, C3, C4, C5]

    def _top_down_pathway(self, conv_list):
        P5 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name='proj_C5',
        )(conv_list[3])
        P4 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name='proj_C4',
        )(conv_list[2])
        P4 = P4 + tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='nearest',
            name='upsampling_P5',
        )(P5)
        P3 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(1, 1), strides=(1, 1), padding="SAME",
            name='proj_C3',
        )(conv_list[1])
        P3 = P3 + tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='nearest',
            name='upsampling_P4',
        )(P4)
        P5 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            name='conv_P5',
        )(P5)
        P4 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            name='conv_P4',
        )(P4)
        P3 = tf.keras.layers.Conv2D(
            filters=self._feature_dimension, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            name='conv_P3',
        )(P3)
        return [P3, P4, P5]

    def call(self, x):
        return self._top_down_pathway(self._bottom_up_pathway(x, block_type='regular', num_blocks_list=[2, 2, 2, 2]))

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="FPN",
        )


if __name__ == "__main__":
    model = FPN(feature_dimension=256)
    model = model.build(input_shape=(512, 512, 3))
    model.summary()
