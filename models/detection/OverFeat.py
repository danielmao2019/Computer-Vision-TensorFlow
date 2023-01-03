import tensorflow as tf


class OverFeat(tf.keras.Model):

    def __init__(self, version, **kwargs):
        if version not in ['fast', 'accurate']:
            raise ValueError(f"[ERROR] version must be in ['fast', 'accurate']. Got {version}.")
        super(OverFeat, self).__init__(name="OverFeat", **kwargs)
        self.version = version

    def _call_fast(self, x):
        x = tf.keras.layers.Conv2D(
            filters=96, kernel_size=(11, 11), strides=(4, 4), padding="VALID", activation='relu',
            name='conv_1',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='max_pool_1',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(5, 5), strides=(1, 1), padding="VALID", activation='relu',
            name='conv_2',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='max_pool_2',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_3',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_4',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_5',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='max_pool_5',
        )(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=3072, name='dense_1',
        )(x)
        x = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_1',
        )(x)
        x = tf.keras.layers.Dense(
            units=4096, name='dense_2',
        )(x)
        x = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_2',
        )(x)
        x = tf.keras.layers.Dense(
            units=1000, name='dense_3',
        )(x)
        return x

    def _call_accurate(self, x):
        x = tf.keras.layers.Conv2D(
            filters=96, kernel_size=(7, 7), strides=(2, 2), padding="VALID", activation='relu',
            name='conv_1',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(3, 3),
            name='max_pool_1',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(7, 7), strides=(1, 1), padding="VALID", activation='relu',
            name='conv_2',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2),
            name='max_pool_2',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_3',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_4',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_5',
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu',
            name='conv_6',
        )(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(3, 3),
            name='max_pool_6',
        )(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=4096, name='dense_1',
        )(x)
        x = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_1',
        )(x)
        x = tf.keras.layers.Dense(
            units=4096, name='dense_2',
        )(x)
        x = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_2',
        )(x)
        x = tf.keras.layers.Dense(
            units=1000, name='dense_3',
        )(x)
        return x

    def call(self, x):
        if self.version == 'fast':
            return self._call_fast(x)
        if self.version == 'accurate':
            return self._call_accurate(x)

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape, name='input_image')
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name=f"OverFeat_{self.version}",
        )


if __name__ == "__main__":
    model_fast = OverFeat(version='fast')
    model_fast = model_fast.build(input_shape=(231, 231, 3))
    model_fast.summary()
    # model_accurate = OverFeat(version='accurate')
    # model_accurate = model_accurate.build(input_shape=(221, 221, 3))
    # model_accurate.summary()
