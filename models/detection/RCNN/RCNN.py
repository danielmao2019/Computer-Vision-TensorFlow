import tensorflow as tf


class RCNN(tf.keras.Model):
    """
    Rich feature hierarchies for accurate object detection and semantic segmentation
    Reference: https://arxiv.org/abs/1311.2524
    """

    def __init__(self, **kwargs):
        super(RCNN, self).__init__(name="RCNN", **kwargs)

    def call(self, x):
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="RCNN",
        )
