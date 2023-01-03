import tensorflow as tf


class SENet(tf.keras.Model):

    def __init__(self, reduction_ratio=16, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self._reduction_ratio = reduction_ratio

    def call(self, x):
        print("In SENet.call():", x.shape)
        identity = x
        height = x.shape[-3]
        width = x.shape[-2]
        channels = x.shape[-1]
        pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        dim_red_layer = tf.keras.layers.Dense(units=channels / self._reduction_ratio)
        dim_inc_layer = tf.keras.layers.Dense(units=channels)
        x = pool_layer(x)
        x = dim_red_layer(x)
        x = tf.keras.activations.relu(x)
        x = dim_inc_layer(x)
        x = tf.keras.activations.sigmoid(x)
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        x = identity * tf.tile(x, multiples=(1, height, width, 1))
        return x

    def build(self, input_shape, name="SENet"):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
            name=name,
        )


if __name__ == "__main__":
    model = SENet()
    model = model.build(input_shape=(224, 224, 256))
    model.summary()
