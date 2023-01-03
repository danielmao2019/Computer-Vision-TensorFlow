import tensorflow as tf
from losses.smooth_l1_loss import SmoothL1Loss


class SSDLoss(tf.losses.Loss):

    def __init__(self, num_classes, delta=1.0):
        """
        Arguments:
            num_classes (int): number of classes in the dataset.
            delta (float): parameter for the localization loss (SmoothL1Loss).
        """
        super(SSDLoss, self).__init__(reduction="auto")
        self._num_classes = num_classes
        self._box_loss = SmoothL1Loss(delta)

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: 3-D tensor of shape (batch_size, num_anchor_boxes, 5).
            y_pred: 3-D tensor of shape (batch_size, num_anchor_boxes, 4 + num_classes).
        Returns:
            total_loss: 1-D tensor of shape (batch_size,).
        """
        assert len(y_true.shape) == 3 and y_true.shape[2] == 5
        assert len(y_pred.shape) == 3 and y_pred.shape[2] == 4 + self._num_classes
        assert y_true.shape[0] == y_pred.shape[0] and y_true.shape[1] == y_pred.shape[1]
        num_anchor_boxes = y_true.shape[1]
        ### Compute the confidence loss
        cls_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE,
        )(y_true[:, :, 4], tf.argmax(y_pred[:, :, 4:], axis=-1))
        positive_mask = tf.cast(tf.equal(y_true[:, :, 4], 0.0), dtype=tf.float32)
        positive_count = tf.reduce_sum(positive_mask, axis=-1)
        positive_loss = tf.where(positive_mask, cls_loss, 0.0)
        positive_loss = tf.reduce_sum(positive_loss, axis=-1)
        negative_mask = tf.logical_not(positive_mask)
        negative_count = tf.reduce_sum(negative_mask, axis=-1)
        negative_loss = tf.where(negative_mask, cls_loss, 0.0)
        negative_loss = tf.nn.top_k(negative_loss, num_anchor_boxes)[0]
        num_max_neg = tf.expand_dims(tf.minimum(negative_count, 3 * positive_count), axis=1)
        range_row = tf.to_int_64(tf.expand_dims(tf.range(0, num_anchor_boxes, 1), axis=0))
        negative_mask = tf.less(range_row, num_max_neg)
        negative_loss = tf.where(negative_mask, negative_loss, 0.0)
        negative_loss = tf.reduce_sum(negative_loss, axis=-1)
        cls_loss = positive_loss + negative_loss
        ### Compute the localization loss
        box_pred = y_pred[:, :, :4]
        box_true = y_true[:, :, :4]
        box_loss = self._box_loss(box_pred, box_true)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        box_loss = tf.reduce_sum(box_loss, axis=-1)
        ### Compute total loss
        total_loss = cls_loss + box_loss
        return total_loss
