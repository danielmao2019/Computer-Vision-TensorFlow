import tensorflow as tf
from sppnet.spatial_pyramid_pool import SpatialPyramidPool


class SPPNet(tf.keras.layers.Layer):

    def __init__(self,
                 input_height, input_width, input_depth,
                 num_classes=10):
        super(SPPNet, self).__init__()
        self.INPUT_HEIGHT = input_height
        self.INPUT_WIDTH = input_width
        self.INPUT_DEPTH = input_depth
        self.NUM_CLASSES = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
        )
        self.conv_2, self.conv_3 = [tf.keras.layers.Conv2D(
            filters=i,
            kernel_size=(3, 3),
            strides=1,
        ) for i in [128, 256]]
        ##################################################
        self.dense_1 = tf.keras.layers.Dense(units=4096)
        self.dense_2 = tf.keras.layers.Dense(
            units=self.NUM_CLASSES,
            activation='softmax',
        )

    def call(self, inputs):
        x = inputs
        for conv_layer in [self.conv_1, self.conv_2, self.conv_3]:
            x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
        spatial_pyramid_pool = SpatialPyramidPool(
            input_height=x.shape[1], input_width=x.shape[2])
        x = spatial_pyramid_pool(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        outputs = x
        return outputs

    def model(self):
        inputs = tf.keras.Input(
            shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_DEPTH))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='SPPNet')
