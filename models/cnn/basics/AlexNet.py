import tensorflow as tf


class AlexNet(tf.keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(AlexNet, self).__init__(name="AlexNet", **kwargs)
        self.INPUT_HEIGHT = 224
        self.INPUT_WIDTH = 224
        self.INPUT_DEPTH = 3
        self.NUM_CLASSES = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=4,
            activation='relu',
        )
        self.max_pool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
        )
        ##################################################
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
        )
        self.max_pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
        )
        ##################################################
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_4 = tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.max_pool_3 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
        )
        ##################################################
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
        )
        self.dropout_1 = tf.keras.layers.Dropout(
            rate=0.2,
        )
        self.dense_2 = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
        )
        self.dropout_2 = tf.keras.layers.Dropout(
            rate=0.2,
        )
        self.dense_3 = tf.keras.layers.Dense(
            units=self.NUM_CLASSES,
            activation='softmax',
        )
        ##################################################

    def call(self, inputs):
        x = inputs
        ##################################################
        x = self.conv_1(x)
        x = self.max_pool_1(x)
        ##################################################
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        ##################################################
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.max_pool_3(x)
        ##################################################
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x, training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training)
        x = self.dense_3(x)
        ##################################################
        outputs = x
        return outputs

    def model(self):
        inputs = tf.keras.Input(
            shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_DEPTH))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='AlexNet')
