import tensorflow as tf


class AuxiliaryNetwork(tf.keras.layers.Layer):

    def __init__(self, num_classes):
        super(AuxiliaryNetwork, self).__init__()
        self.NUM_CLASSES = num_classes
        self.avg_pooling = tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5),
            strides=3,
        )
        self.conv = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(1, 1),
            padding='same',
            activation='relu'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            units=1024,
            activation='relu',
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.7)
        self.dense_2 = tf.keras.layers.Dense(
            units=self.NUM_CLASSES,
            activation='softmax',
        )

    def call(self, inputs):
        x = inputs
        x = self.avg_pooling(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        outputs = x
        return outputs
