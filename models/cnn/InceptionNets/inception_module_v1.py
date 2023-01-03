import tensorflow as tf


class InceptionModuleV1(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(InceptionModuleV1, self).__init__(**kwargs)
        ##################################################
        self.conv_1x1 = tf.keras.layers.Conv2D(
            filters=filters['conv_1x1'],
            kernel_size=1,
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_3x3_reduction = tf.keras.layers.Conv2D(
            filters=filters['conv_3x3_reduction'],
            kernel_size=1,
            padding='same',
            activation='relu',
        )
        self.conv_3x3 = tf.keras.layers.Conv2D(
            filters=filters['conv_3x3'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.conv_5x5_reduction = tf.keras.layers.Conv2D(
            filters=filters['conv_5x5_reduction'],
            kernel_size=1,
            padding='same',
            activation='relu',
        )
        self.conv_5x5 = tf.keras.layers.Conv2D(
            filters=filters['conv_5x5'],
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
        )
        ##################################################
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=1,
            padding='same',
        )
        self.max_pool_projection = tf.keras.layers.Conv2D(
            filters=filters['max_pool_projection'],
            kernel_size=1,
            padding='same',
            activation='relu',
        )
        ##################################################

    def call(self, inputs):
        ##################################################
        conv_1x1 = self.conv_1x1(inputs)
        ##################################################
        conv_3x3 = self.conv_3x3_reduction(inputs)
        conv_3x3 = self.conv_3x3(conv_3x3)
        ##################################################
        conv_5x5 = self.conv_5x5_reduction(inputs)
        conv_5x5 = self.conv_5x5(conv_5x5)
        ##################################################
        max_pool = self.max_pool(inputs)
        max_pool = self.max_pool_projection(max_pool)
        ##################################################
        outputs = tf.concat([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)
        return outputs
