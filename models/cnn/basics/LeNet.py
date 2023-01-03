import tensorflow as tf


class LeNet(tf.keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(LeNet, self).__init__(name="LeNet", **kwargs)
        self.INPUT_HEIGHT = 32
        self.INPUT_WIDTH = 32
        self.INPUT_DEPTH = 1
        self.NUM_CLASSES = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='tanh',
        )
        self.avg_pooling_1 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
        )
        ##################################################
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='tanh',
        )
        self.avg_pooling_2 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
        )
        ##################################################
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=120,
            kernel_size=(5, 5),
            strides=1,
            padding='valid',
            activation='tanh',
        )
        ##################################################
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            units=84,
            activation='tanh',
        )
        self.dense_2 = tf.keras.layers.Dense(
            units=self.NUM_CLASSES,
            activation='softmax',
        )
        ##################################################

    def call(self, inputs):
        x = inputs
        ##################################################
        x = self.conv_1(x)
        x = self.avg_pooling_1(x)
        ##################################################
        x = self.conv_2(x)
        x = self.avg_pooling_2(x)
        ##################################################
        x = self.conv_3(x)
        ##################################################
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        ##################################################
        outputs = x
        return outputs

    def model(self):
        inputs = tf.keras.Input(
            shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_DEPTH))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='LeNet')
