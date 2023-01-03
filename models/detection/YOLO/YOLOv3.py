import tensorflow as tf


class YOLOv3Backbone(tf.keras.Model):

    def __init__(self, **kwargs):
        super(YOLOv3Backbone, self).__init__(name="YOLOv3Backbone", **kwargs)

    def _conv_block(self, x, filters, kernel_size, strides=(1, 1)):
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _residual_block(self, x, filters):
        shortcut = x
        x = self._conv_block(x, filters=filters, kernel_size=(1, 1))
        x = self._conv_block(x, filters=filters * 2, kernel_size=(3, 3))
        x = tf.keras.layers.Add([x, shortcut])
        return x

    def call(self, x):
        x = self._conv_block(x, filters=32, kernel_size=(3, 3))
        x = self._conv_block(x, filters=64, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(1):
            x = self._residual_block(x, filters=32)
        x = self._conv_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(2):
            x = self._residual_block(x, filters=64)
        x = self._conv_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(8):
            x = self._residual_block(x, filters=128)
        route1 = x
        x = self._conv_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(8):
            x = self._residual_block(x, filters=256)
        route2 = x
        x = self._conv_block(x, filters=1024, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(4):
            x = self._residual_block(x, filters=512)
        route3 = x
        return route1, route2, route3


class YOLOv3(tf.keras.Model):

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(YOLOv3, self).__init__(name="YOLOv3", **kwargs)
        self._num_classes = num_classes
        if not backbone:
            backbone = YOLOv3Backbone()
        self._backbone = backbone

    def _conv_block(self, x, filters, kernel_size, strides=(1, 1)):
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _head(self, x, filters):
        for _ in range(2):
            x = self._conv_block(x, filters=filters, kernel_size=(1, 1))
            x = self._conv_block(x, filters=filters * 2, kernel_size=(3, 3))
        x = self._conv_block(x, filters=filters, kernel_size=(1, 1))
        boxes = self._conv_block(x, filters=filters * 2, kernel_size=(3, 3))
        boxes = tf.keras.layers.Conv2D(
            filters=3 * (self._num_classes + 5), kernel_size=(1, 1), padding="same",
        )(boxes)
        x = self._conv_block(x, filters=filters // 2, kernel_size=(1, 1))
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters // 2, kernel_size=(2, 2), strides=(2, 2), padding="same",
        )(x)
        return x, boxes

    def call(self, x):
        route1, route2, route3 = self._backbone(x)
        x = route3
        x, boxes1 = self._head(x, filters=512)
        x = tf.concat([x, route2], axis=-1)
        x, boxes2 = self._head(x, filters=256)
        x = tf.concat([x, route1], axis=-1)
        x, boxes3 = self._head(x, filters=128)
        return boxes1, boxes2, boxes3
