import tensorflow as tf
from classification.models.cnn.residual_block import *


class ResNet(tf.keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(ResNet, self).__init__(name="ResNet", **kwargs)
        self._input_height = 224
        self._input_width = 224
        self._input_depth = 3
        self._num_classes = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
        )
        self.max_pool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
        ##################################################
        self.residual_block_2a = ResidualBlock(
            in_channels=128,
            out_channels=256,
            project_shortcut=True,
            name='residual_block_2a',
        )
        self.residual_block_2b, self.residual_block_2c = [ResidualBlock(
            in_channels=128,
            out_channels=256,
            name=name,
        ) for name in ['residual_block_2b',
                       'residual_block_2c',
                       ]]
        ##################################################
        self.residual_block_3a = ResidualBlock(
            in_channels=256,
            out_channels=512,
            strides=(2, 2),
            name='residual_block_3a',
        )
        self.residual_block_3b, self.residual_block_3c, self.residual_block_3d = [ResidualBlock(
            in_channels=256,
            out_channels=512,
            name=name,
        ) for name in ['residual_block_3b',
                       'residual_block_3c',
                       'residual_block_3d',
                       ]]
        ##################################################
        self.residual_block_4a = ResidualBlock(
            in_channels=512,
            out_channels=1024,
            strides=(2, 2),
            name='residual_block_4a',
        )
        self.residual_block_4b, self.residual_block_4c, self.residual_block_4d, self.residual_block_4e, self.residual_block_4f = [ResidualBlock(
            in_channels=512,
            out_channels=1024,
            name=name,
        ) for name in ['residual_block_4b',
                       'residual_block_4c',
                       'residual_block_4d',
                       'residual_block_4e',
                       'residual_block_4f',
                       ]]
        ##################################################
        self.residual_block_5a = ResidualBlock(
            in_channels=1024,
            out_channels=2048,
            strides=(2, 2),
            name='residual_block_5a',
        )
        self.residual_block_5b, self.residual_block_5c = [ResidualBlock(
            in_channels=1024,
            out_channels=2048,
            name=name,
        ) for name in ['residual_block_5b',
                       'residual_block_5c',
                       ]]
        ##################################################
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(units=self._num_classes)
        ##################################################

    def call(self, x):
        ##################################################
        x = self.conv_1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.max_pool_1(x)
        ##################################################
        x = self.residual_block_2a(x)
        x = self.residual_block_2b(x)
        x = self.residual_block_2c(x)
        ##################################################
        x = self.residual_block_3a(x)
        x = self.residual_block_3b(x)
        x = self.residual_block_3c(x)
        x = self.residual_block_3d(x)
        ##################################################
        x = self.residual_block_4a(x)
        x = self.residual_block_4b(x)
        x = self.residual_block_4c(x)
        x = self.residual_block_4d(x)
        x = self.residual_block_4e(x)
        x = self.residual_block_4f(x)
        ##################################################
        x = self.residual_block_5a(x)
        x = self.residual_block_5b(x)
        x = self.residual_block_5c(x)
        ##################################################
        x = self.avg_pooling(x)
        x = self.dense(x)
        ##################################################
        return x

    def model(self):
        inputs = tf.keras.Input(
            shape=(self._input_height, self._input_width, self._input_depth))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='ResNet50')
