import tensorflow as tf


class MixerBlock(tf.keras.layers.Layer):

    def __init__(self,
                 num_patches, embedding_dim,
                 dropout=True, dropout_rate=0.5,
                 *args, **kwargs):
        """__init__ method
        
        Arguments:
            num_patches (int): the number of patches each image is extracted into.
            embedding_dim (int): the dimension of the space the patches are embedded into.
            dropout (bool): the controller of whether of not to use dropout layers.
            dropout_rate (float): the rate for the dropout layer, if any.
        """
        super(MixerBlock, self).__init__(name="Mixer Block", *args, **kwargs)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.MLPBlock1 = MLPBlock(num_patches, num_patches, dropout, dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.MLPBlock2 = MLPBlock(num_patches, embedding_dim, dropout, dropout_rate)

    def call(self, x, training=False):
        ### token-mixer
        resid = x
        x = self.layernorm1(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.MLPBlock1(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x += resid
        ### channel-mixer
        resid = x
        x = self.layernorm2(x)
        x = self.MLPBlock2(x)
        x += resid
        return x


class MLPBlock(tf.keras.layers.Layer):

    def __init__(self,
                 units1, units2,
                 dropout=True, dropout_rate=0.5,
                 *args, **kwargs):
        """__init__ method
        
        Arguments:
            units1 (int): the number of units in the first Dense layer.
            units2 (int): the number of units in the second Dense layer.
            dropout (bool): the controller of whether of not to use dropout layers.
            dropout_rate (float): the rate for the dropout layer, if any.
        """
        super(MLPBlock, self).__init__(name="MLP Block", *args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(units=units1)
        self.dense2 = tf.keras.layers.Dense(units=units2)
        if dropout:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.keras.activations.gelu(x)
        x = self.dense2(x)
        if self.dropout:
            x = self.dropout(x)
        return x
