import tensorflow as tf
import math


class SpatialPyramidPool(tf.keras.layers.Layer):

    def __init__(self, input_height, input_width):
        super(SpatialPyramidPool, self).__init__()
        self.INPUT_HEIGHT = input_height
        self.INPUT_WIDTH = input_width
        ##################################################
        max_pool_layers = []
        for i in [4, 2, 1]:
            kernel_height = int(math.ceil(self.INPUT_HEIGHT/i))
            kernel_width = int(math.ceil(self.INPUT_WIDTH/i))
            padding_height = int((kernel_height*i-self.INPUT_HEIGHT+1)/2)
            padding_width = int((kernel_width*i-self.INPUT_WIDTH+1)/2)
            layer = tf.keras.layers.MaxPool2D(
                pool_size=(kernel_height, kernel_width),
                strides=(kernel_height, kernel_width),
                padding='same',
                # padding=(padding_height, padding_width),
                name='max_pool_'+str(i)+'x'+str(i),
            )
            max_pool_layers.append(layer)
        self.max_pool_4x4, self.max_pool_2x2, self.max_pool_1x1 = max_pool_layers
        ##################################################
        self.flatten = tf.keras.layers.Flatten()
        ##################################################

    def call(self, inputs):
        max_pool_4x4 = self.max_pool_4x4(inputs)
        max_pool_4x4 = self.flatten(max_pool_4x4)
        max_pool_2x2 = self.max_pool_2x2(inputs)
        max_pool_2x2 = self.flatten(max_pool_2x2)
        max_pool_1x1 = self.max_pool_1x1(inputs)
        max_pool_1x1 = self.flatten(max_pool_1x1)
        ##################################################
        outputs = tf.concat(
            [max_pool_4x4, max_pool_2x2, max_pool_1x1], axis=-1)
        return outputs
