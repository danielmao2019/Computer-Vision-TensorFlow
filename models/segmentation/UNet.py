import tensorflow as tf


class UNet(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(UNet, self).__init__(name="UNet", **kwargs)
        self._num_classes = num_classes

    def _downsampling_block(self, x, filters, block_id):
        for it in range(2):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                name=block_id + f"_conv_{it+1}",
            )(x)
            x = tf.keras.layers.ReLU(
                name=block_id + f"_relu_{it+1}",
            )(x)
        skip = x
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name=block_id + '_max_pool',
        )(x)
        return x, skip

    def _upsampling_block(self, x, skip, filters, block_id):
        for it in range(2):
            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                name=block_id + f"_conv_{it+1}",
            )(x)
            x = tf.keras.layers.ReLU(
                name=block_id + f"_relu_{it+1}",
            )(x)
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear",
            name=block_id + "_upsampling",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=filters//2, kernel_size=(2, 2), strides=(1, 1), padding="SAME",
            name=block_id + "_up_conv",
        )(x)
        height_diff = skip.shape[1] - x.shape[1]
        width_diff = skip.shape[2] - x.shape[2]
        skip = tf.keras.layers.Cropping2D(
            cropping=((height_diff//2, height_diff - height_diff//2), (width_diff//2, width_diff - width_diff//2)),
            name=block_id + "_crop",
        )(skip)
        x = tf.keras.layers.Concatenate(
            name=block_id + "_concat",
        )([x, skip])
        return x

    def _contracting_path(self, x):
        x, skip0 = self._downsampling_block(x, filters=64, block_id='contracting_1')
        x, skip1 = self._downsampling_block(x, filters=128, block_id='contracting_2')
        x, skip2 = self._downsampling_block(x, filters=256, block_id='contracting_3')
        x, skip3 = self._downsampling_block(x, filters=512, block_id='contracting_4')
        return x, [skip0, skip1, skip2, skip3]

    def _expansive_path(self, x, skip_list):
        x = self._upsampling_block(x, skip_list[3], filters=1024, block_id='expansive_1')
        x = self._upsampling_block(x, skip_list[2], filters=512, block_id='expansive_2')
        x = self._upsampling_block(x, skip_list[1], filters=256, block_id='expansive_3')
        x = self._upsampling_block(x, skip_list[0], filters=128, block_id='expansive_4')
        return x

    def call(self, x):
        x, skip_list = self._contracting_path(x)
        x = self._expansive_path(x, skip_list)
        for it in range(2):
            x = tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                name=f"final_conv_{it+1}",
            )(x)
            x = tf.keras.layers.ReLU(
                name=f"final_relu_{it+1}",
            )(x)
        x = tf.keras.layers.Conv2D(
            filters=self._num_classes, kernel_size=(1, 1), strides=(1, 1), padding="VALID",
            name="final_projection",
        )(x)
        x = tf.keras.layers.Softmax()(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="UNet",
        )


if __name__ == "__main__":
    model = UNet(num_classes=2)
    model = model.build(input_shape=(572, 572, 3))
    model.summary()
