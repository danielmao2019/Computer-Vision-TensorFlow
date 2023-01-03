import tensorflow as tf


class AlexNet(tf.keras.Model):
    """
    ImageNet Classification with Deep Convolutional Neural Networks
    Reference: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
    """

    def __init__(self, num_classes=10, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        self.conv_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        self.conv_3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense_2 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.5)
        self.dense_3 = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, x, training=None):
        x = self.conv_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.max_pool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_3(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name='AlexNet',
        )
