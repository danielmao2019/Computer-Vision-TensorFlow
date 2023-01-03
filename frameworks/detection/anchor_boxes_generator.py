import tensorflow as tf


class AnchorBoxesGenerator:
    """
    Anchor boxes generated are in the format 'center'.
    """

    def __init__(self,
                 ratios=[0.5, 1.0, 2.0],
                 scales=[2 ** x for x in [0, 1 / 3, 2 / 3]],
                 strides=[2 ** i for i in range(3, 8)],
                 areas=[x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]):
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        self.areas = areas
        self.num_anchor_boxes = len(ratios) * len(scales)
        self.anchor_box_dims = self.get_anchor_box_dims()

    def get_anchor_box_dims(self):
        """
        Returns a tensor of shape (num_areas, num_ratios * num_scales, 1, 1, 2).
        """
        all_dims = []
        for area in self.areas:
            dims_for_the_level = []
            for ratio in self.ratios:
                height = tf.math.sqrt(area / ratio)
                width = area / height
                dims = tf.stack([width, height], axis=-1)
                for scale in self.scales:
                    dims_for_the_level.append(dims * scale)
            all_dims.append(tf.stack(dims_for_the_level, axis=0))
        return tf.stack(all_dims, axis=0)

    def get_anchor_boxes_for_the_level(self, feature_height, feature_width, level):
        centers_xcoord = tf.range(feature_width, dtype=tf.float32) + 0.5
        centers_ycoord = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(centers_xcoord, centers_ycoord), axis=-1) * self.strides[level-3]
        centers = tf.tile(centers[:, :, None, :], multiples=[1, 1, self.num_anchor_boxes, 1])
        dims = tf.tile(self.anchor_box_dims[level-3, None, None, :, :], multiples=[feature_height, feature_width, 1, 1])
        anchor_boxes = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchor_boxes, shape=[-1, 4])

    def get_anchor_boxes(self, image_height, image_width):
        assert (image_height is not None) and (image_width is not None)
        anchors = [self.get_anchor_boxes_for_the_level(
                    feature_height=tf.math.ceil(image_height / 2 ** i),
                    feature_width=tf.math.ceil(image_width / 2 ** i),
                    level=i) for i in range(3, 8)]
        return tf.concat(anchors, axis=0)
