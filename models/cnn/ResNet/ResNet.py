import tensorflow as tf


class ResNet(tf.keras.Model):
    """
    Deep Residual Learning for Image Recognition
    Reference: https://arxiv.org/abs/1512.03385
    """

    VERSION_OPTIONS = [18, 34, 50, 101, 152]

    def __init__(self, num_classes, version, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self._num_classes = num_classes
        if version not in self.VERSION_OPTIONS:
            raise ValueError(f"[ERROR] Argument \'version\' must be in {self.VERSION_OPTIONS}. Got {version}.")
        self.version = version

    def _conv_layer(self, x, filters, strides, layer_id):
        """
        Perform 3x3 convolution + batch normalization + ReLU activation on input tensor x.
        Strides of the convolution operation can vary.
        Arguments:
            filters (int): number of filters for the 3x3 convolution.
            layer_id (str): id of the current layer.
        Returns:
            x (tf.Tensor): processed input.
        """
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
        """
        Perform 1x1 convolution + batch normalization + ReLU activation on input tensor x to decrease dimension.
        Strides of the convolution operation is fixed to (1, 1).
        """
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
        """
        Perform 1x1 convolution + batch normalization on input tensor x to increase dimension.
        Strides of the convolution operation is fixed to (1, 1).
        """
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
            strides (int): strides for the 3x3 convolution in the first self._conv_layer.
        """
        shortcut = x
        x = self._conv_layer(x, filters=filters, strides=strides, layer_id=block_id + '_layer_1')
        x = self._conv_layer(x, filters=filters, strides=(1, 1), layer_id=block_id + '_layer_2')
        if x.shape[3] != shortcut.shape[3]:
            assert x.shape[3] == shortcut.shape[3] * 2
            assert shortcut.shape[1] == x.shape[1] * 2
            assert shortcut.shape[2] == x.shape[2] * 2
            assert strides in [2, (2, 2)]
            shortcut = tf.keras.layers.Conv2D(
                filters=x.shape[3], kernel_size=(1, 1), strides=strides, padding="SAME",
                name=block_id+'_projection',
            )(shortcut)
        x = tf.keras.layers.Add(
            name=block_id+'_merge',
        )([x, shortcut])
        x = tf.keras.layers.ReLU(
            name=block_id+'_final_relu',
        )(x)
        return x

    def _bottleneck_block(self, x, filters, strides, block_id):
        """
        Arguments:
            filters (int): number of filters for reduction and conv. The number of filters for
                           expansion is 4 * filters.
        """
        shortcut = x
        x = self._reduction_layer(x, filters=filters, layer_id=block_id + '_layer_1')
        x = self._conv_layer(x, filters=filters, strides=strides, layer_id=block_id + '_layer_2')
        x = self._expansion_layer(x, filters=filters*4, layer_id=block_id + '_layer_3')
        if x.shape[3] != shortcut.shape[3]:
            assert x.shape[3] == shortcut.shape[3] * 2
            assert shortcut.shape[1] == x.shape[1] * 2
            assert shortcut.shape[2] == x.shape[2] * 2
            assert strides in [2, (2, 2)]
            shortcut = tf.keras.layers.Conv2D(
                filters=x.shape[3], kernel_size=(1, 1), strides=strides, padding="SAME",
                name=block_id+'_projection',
            )(shortcut)
        x = tf.keras.layers.Add(
            name=block_id+'_merge',
        )([x, shortcut])
        x = tf.keras.layers.ReLU(
            name=block_id+'_final_relu',
        )(x)
        return x

    def _call_stage(self, x, block_type, filters, first_strides, num_blocks, stage_id):
        """
        Arguments:
            block_type (str): The uniform type of the block to be used in each stage. Should be one of ['regular', 'bottleneck'].
            filters (int): The uniform number of filters for the convolution layers in each of the blocks.
            first_strides (int): strides of the convolution in the first self._conv_layer in the first block.
            num_blocks (int): number of times the block is repeated.
        """
        if block_type not in ['regular', 'bottleneck']:
            raise ValueError(f"[ERROR] block_type must be in ['regular', 'bottleneck']. Got {block_type}.")
        block = self._regular_block if block_type == 'regular' else self._bottleneck_block
        for idx in range(num_blocks):
            strides = first_strides if idx == 0 else (1, 1)
            x = block(x, filters=filters, strides=strides, block_id=stage_id+f"b{idx+1}")
        return x

    def _classifier(self, x):
        x = tf.keras.layers.GlobalAveragePooling2D(
            name='final_global_average_pooling',
        )(x)
        x = tf.keras.layers.Dense(
            units=self._num_classes,
            name='final_dense_layer',
        )(x)
        x = tf.keras.layers.Softmax(
            name='final_softmax_layer',
        )(x)
        return x

    def _call_version(self, x, block_type, num_blocks_list):
        """
        Arguments:
            block_type (str): The uniform type of the block to be used in each stage. Should be one of ['regular', 'bottleneck'].
            num_blocks_list (list of int): The number of blocks in each stage.
        """
        assert len(num_blocks_list) == 4
        input_h, input_w = x.shape[-3], x.shape[-2]
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME",
            name='stage_1_conv',
        )(x)
        assert x.shape[-3] == input_h // 2 and x.shape[-2] == input_w // 2
        x = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding="SAME",
            name='stage_2_max_pool',
        )(x)
        x = self._call_stage(
            x, block_type=block_type, filters=64, first_strides=(1, 1),
            num_blocks=num_blocks_list[0], stage_id="s2",
        )
        assert x.shape[-3] == input_h // 4 and x.shape[-2] == input_w // 4
        x = self._call_stage(
            x, block_type=block_type, filters=128, first_strides=(2, 2),
            num_blocks=num_blocks_list[1], stage_id="s3",
        )
        assert x.shape[-3] == input_h // 8 and x.shape[-2] == input_w // 8
        x = self._call_stage(
            x, block_type=block_type, filters=256, first_strides=(2, 2),
            num_blocks=num_blocks_list[2], stage_id="s4",
        )
        assert x.shape[-3] == input_h // 16 and x.shape[-2] == input_w // 16
        x = self._call_stage(
            x, block_type=block_type, filters=512, first_strides=(2, 2),
            num_blocks=num_blocks_list[3], stage_id="s5",
        )
        assert x.shape[-3] == input_h // 32 and x.shape[-2] == input_w // 32
        x = self._classifier(x)
        return x

    def call(self, x):
        if self.version == 18:
            x = self._call_version(x, block_type='regular', num_blocks_list=[2, 2, 2, 2])
        elif self.version == 34:
            x = self._call_version(x, block_type='regular', num_blocks_list=[3, 4, 6, 3])
        elif self.version == 50:
            x = self._call_version(x, block_type='bottleneck', num_blocks_list=[3, 4, 6, 3])
        elif self.version == 101:
            x = self._call_version(x, block_type='bottleneck', num_blocks_list=[3, 4, 23, 3])
        elif self.version == 152:
            x = self._call_version(x, block_type='bottleneck', num_blocks_list=[3, 8, 36, 3])
        else:
            assert False
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name=f"ResNet{self.version}",
        )
