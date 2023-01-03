import tensorflow as tf
from SENet import SENet


class MobileNetV3(tf.keras.Model):
    """
    Reference: https://arxiv.org/abs/1905.02244.
    """

    def __init__(self, model_type, output_dim, **kwargs):
        """
        Arguments:
            model_type (str): one of ['large', 'small'].
        """
        super(MobileNetV3, self).__init__(name="MobileNetV3", **kwargs)
        self._model_type = model_type
        self._output_dim = output_dim

    def _hard_swish(self, x):
        return x * tf.keras.activations.relu(x + 3, max_value=6) / 6

    def _bottleneck_block(self, x, exp, out, kernel_size, strides, use_SE, nl):
        """
        Arguments:
            x (tensor): input to the bottleneck block.
            exp (int): number of channels of the expansion layer.
            out (int): number of channels of the outputs.
            kernel_size (int): kernel size of the depthwise convolution.
            strides (int): strides of the depthwise convolution.
            use_SE (bool): whether or not to use Squeeze-And-Excite.
            nl (function): non-linear activation function.
        """
        shortcut = x
        exp_conv = tf.keras.layers.Conv2D(
            filters=exp, kernel_size=1, strides=1, padding="same",
        )
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding="same",
        )
        out_conv = tf.keras.layers.Conv2D(
            filters=out, kernel_size=1, strides=1, padding="same",
        )
        x = exp_conv(x)
        x = nl(x)
        x = depthwise_conv(x)
        x = nl(x)
        x = out_conv(x)
        print("In MobileNetV3.call():", x.shape)
        if use_SE:
            x = SENet().build(input_shape=x.shape[1:], name=str(exp)+str(out))(x)
        if x.shape == shortcut.shape:
            x = tf.keras.layers.Add()([x, shortcut])
        return x

    def _call_large(self, x):
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=2, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self._hard_swish(x)
        x = self._bottleneck_block(x, exp=16, out=16, kernel_size=3, strides=1, use_SE=False, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=64, out=24, kernel_size=3, strides=2, use_SE=False, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=72, out=24, kernel_size=3, strides=1, use_SE=False, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=72, out=40, kernel_size=5, strides=2, use_SE=True, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=120, out=40, kernel_size=5, strides=1, use_SE=True, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=120, out=40, kernel_size=5, strides=1, use_SE=True, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=240, out=80, kernel_size=3, strides=2, use_SE=False, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=200, out=80, kernel_size=3, strides=1, use_SE=False, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=184, out=80, kernel_size=3, strides=1, use_SE=False, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=184, out=80, kernel_size=3, strides=1, use_SE=False, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=480, out=112, kernel_size=3, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=672, out=112, kernel_size=3, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=672, out=160, kernel_size=5, strides=2, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=960, out=160, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=960, out=160, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = tf.keras.layers.Conv2D(
            filters=960, kernel_size=1, strides=1, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self._hard_swish(x)
        x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=1, strides=1, padding="valid",
        )(x)
        x = self._hard_swish(x)
        x = tf.keras.layers.Conv2D(
            filters=self._output_dim, kernel_size=1, strides=1, padding="valid",
        )(x)
        return x

    def _call_small(self, x):
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=2, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self._hard_swish(x)
        x = self._bottleneck_block(x, exp=16, out=16, kernel_size=3, strides=2, use_SE=True, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=72, out=24, kernel_size=3, strides=2, use_SE=False, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=88, out=24, kernel_size=3, strides=1, use_SE=False, nl=tf.keras.activations.relu)
        x = self._bottleneck_block(x, exp=96, out=40, kernel_size=5, strides=2, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=240, out=40, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=240, out=40, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=120, out=48, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=144, out=48, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=288, out=96, kernel_size=5, strides=2, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=576, out=96, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = self._bottleneck_block(x, exp=576, out=96, kernel_size=5, strides=1, use_SE=True, nl=self._hard_swish)
        x = tf.keras.layers.Conv2D(
            filters=576, kernel_size=1, strides=1, padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # ??? not sure. how about SE added to a single conv layer? where should I put the activation?
        x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=1, strides=1, padding="valid",
        )(x)
        x = self._hard_swish(x)
        x = tf.keras.layers.Conv2D(
            filters=self._output_dim, kernel_size=1, strides=1, padding="valid",
        )(x)
        return x

    def call(self, x):
        if self._model_type == 'large':
            return self._call_large(x)
        elif self._model_type == 'small':
            return self._call_small(x)
        else:
            raise ValueError(f"[ERROR] Not recognized model type {self._model_type}.")

    def build(self, input_shape=(224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs,
            name="MobileNetV3_" + self._model_type,
        )


if __name__ == "__main__":
    model = MobileNetV3(model_type='small', output_dim=10)
    model = model.build()
    model.summary()
