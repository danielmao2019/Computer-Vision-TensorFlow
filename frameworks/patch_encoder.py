import tensorflow as tf


class PatchEncoder(tf.keras.layers.Layer):

    def __init__(self, patch_size, num_patches):
        """__init__ method
        
        Arguments:
            patch_size (int): side length of the square patches the image is extracted into.
            num_patches (int): the number of patches the image is extracted into.
        """
        super(PatchEncoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches
