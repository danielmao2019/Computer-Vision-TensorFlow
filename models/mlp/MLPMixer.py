import tensorflow as tf
from classification.models.mlp.mixer_block import *


class MLPMixer(tf.keras.Model):

    def __init__(self,
                 num_classes, num_mixer_blocks,
                 num_patches, embedding_dim,
                 dropout=True, dropout_rate=0.5,
                 *args, **kwargs):
        """__init__ method
        
        Arguments:
            num_classes (int): the number of classes in the dataset.
            num_mixer_blocks (int): the number of Mixer blocks in the MLPMixer model.
            num_patches (int): the number of patches each image is extracted into.
            embedding_dim (int): the dimension of the space the patches are embedded into.
            dropout (bool): the controller of whether of not to use dropout layers.
            dropout_rate (float): the rate for the dropout layer, if any.
        """
        super(MLPMixer, self).__init__(name="MLP Mixer", *args, **kwargs)
        self.embed = tf.keras.layers.Dense(units=embedding_dim)
        self.mixer_blocks = [MixerBlock(num_patches, embedding_dim, dropout, dropout_rate)
                             for _ in range(num_mixer_blocks)]
        self.dense = tf.keras.layers.Dense(units=num_classes, activation="softmax")
        if dropout:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None

    def call(self, x, training=False):
        x = self.embed(x)
        for block in self.mixer_blocks:
            x = block(x)
        x = tf.keras.layers.GlobalAveragePooling1D(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.dense(x)
        return x
