import tensorflow as tf
from InceptionNetworks.InceptionV1.inception_module_v1 import InceptionModuleV1
from InceptionNetworks.InceptionV1.auxiliary_network import AuxiliaryNetwork


class InceptionV1(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(InceptionV1, self).__init__()
        self.INPUT_HEIGHT = 229
        self.INPUT_WIDTH = 229
        self.INPUT_DEPTH = 3
        self.NUM_CLASSES = num_classes
        ##################################################
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            activation='relu',
            name='conv_1',
        )
        self.max_pool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding='same',
            name='max_pool_1',
        )
        ##################################################
        self.conv_2_reduction = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=1,
            activation='relu',
            name='conv_2_reduction',
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            name='conv_2',
        )
        self.max_pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding='same',
            name='max_pool_2',
        )
        ##################################################
        self.inception_3a = InceptionModuleV1(filters={
            'conv_1x1': 64,
            'conv_3x3_reduction': 96,
            'conv_3x3': 128,
            'conv_5x5_reduction': 16,
            'conv_5x5': 32,
            'max_pool_projection': 32,
        }, name='inception_3a')
        self.inception_3b = InceptionModuleV1(filters={
            'conv_1x1': 128,
            'conv_3x3_reduction': 128,
            'conv_3x3': 192,
            'conv_5x5_reduction': 32,
            'conv_5x5': 96,
            'max_pool_projection': 64,
        }, name='inception_3b')
        self.max_pool_3 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding='same',
            name='max_pool_3',
        )
        ##################################################
        self.inception_4a = InceptionModuleV1(filters={
            'conv_1x1': 192,
            'conv_3x3_reduction': 96,
            'conv_3x3': 208,
            'conv_5x5_reduction': 16,
            'conv_5x5': 48,
            'max_pool_projection': 64,
        }, name='inception_4a')
        self.aux_1 = AuxiliaryNetwork(self.NUM_CLASSES)
        self.inception_4b = InceptionModuleV1(filters={
            'conv_1x1': 160,
            'conv_3x3_reduction': 112,
            'conv_3x3': 224,
            'conv_5x5_reduction': 24,
            'conv_5x5': 64,
            'max_pool_projection': 64,
        }, name='inception_4b')
        self.inception_4c = InceptionModuleV1(filters={
            'conv_1x1': 128,
            'conv_3x3_reduction': 128,
            'conv_3x3': 256,
            'conv_5x5_reduction': 24,
            'conv_5x5': 64,
            'max_pool_projection': 64,
        }, name='inception_4c')
        self.inception_4d = InceptionModuleV1(filters={
            'conv_1x1': 112,
            'conv_3x3_reduction': 144,
            'conv_3x3': 288,
            'conv_5x5_reduction': 32,
            'conv_5x5': 64,
            'max_pool_projection': 64,
        }, name='inception_4d')
        self.aux_2 = AuxiliaryNetwork(self.NUM_CLASSES)
        self.inception_4e = InceptionModuleV1(filters={
            'conv_1x1': 256,
            'conv_3x3_reduction': 160,
            'conv_3x3': 320,
            'conv_5x5_reduction': 32,
            'conv_5x5': 128,
            'max_pool_projection': 128,
        }, name='inception_4e')
        self.max_pool_4 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding='same',
            name='max_pool_4',
        )
        ##################################################
        self.inception_5a = InceptionModuleV1(filters={
            'conv_1x1': 256,
            'conv_3x3_reduction': 160,
            'conv_3x3': 320,
            'conv_5x5_reduction': 32,
            'conv_5x5': 128,
            'max_pool_projection': 128,
        }, name='inception_5a')
        self.inception_5b = InceptionModuleV1(filters={
            'conv_1x1': 384,
            'conv_3x3_reduction': 192,
            'conv_3x3': 384,
            'conv_5x5_reduction': 48,
            'conv_5x5': 128,
            'max_pool_projection': 128,
        }, name='inception_5b')
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        ##################################################
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.dense = tf.keras.layers.Dense(
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
        x = self.conv_2_reduction(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        ##################################################
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)
        ##################################################
        x = self.inception_4a(x)
        aux_1 = self.aux_1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux_2 = self.aux_2(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        ##################################################
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pooling(x)
        ##################################################
        x = self.dropout(x)
        x = self.dense(x)
        ##################################################
        outputs = [x, aux_1, aux_2]
        return outputs

    def model(self):
        inputs = tf.keras.Input(
            shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH, self.INPUT_DEPTH))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
                              name='InceptionV1')
