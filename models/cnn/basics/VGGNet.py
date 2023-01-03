import tensorflow as tf


class VGGNet(tf.keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(VGGNet, self).__init__(name="VGG", **kwargs)
        self.INPUT_HEIGHT = 224
        self.INPUT_WIDTH = 224
        self.INPUT_DEPTH = 3
        self.NUM_CLASSES = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_4 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_6 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_7 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_8 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_9 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_10 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_11 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_12 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        self.conv_13 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
        )
        self.dense_2 = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
        )
        self.dense_3 = tf.keras.layers.Dense(
            units=self.NUM_CLASSES,
            activation='softmax',
        )
        ##################################################

    def call(self, x):
        max_pool_layer = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
        )
        ##################################################
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = max_pool_layer(x)
        ##################################################
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = max_pool_layer(x)
        ##################################################
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = max_pool_layer(x)
        ##################################################
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        x = max_pool_layer(x)
        ##################################################
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = max_pool_layer(x)
        ##################################################
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        ##################################################
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='VGGNet',
                              )


if __name__ == "__main__":
    model = VGGNet()
    model = model.build()
    print(model.summary())
