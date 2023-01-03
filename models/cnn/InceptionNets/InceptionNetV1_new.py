import tensorflow as tf


class InceptionNetV1(tf.keras.Model):

    def __init__(self, **kwargs):
        super(InceptionNetV1, self).__init__(name="InceptionNetV1", **kwargs)

    def call(self, x):
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
            name="InceptionNetV1",
        )


if __name__ == "__main__":
    model = InceptionNetV1()
    model = model.build()
    model.summary()
