import tensorflow as tf
from losses.focal_loss import *
from losses.smooth_l1_loss import *


class RetinaNetLoss(tf.losses.Loss):

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        """
        Arguments:
            num_classes (int): number of classes in the dataset.
            alpha (float): parameter for the classification loss (FocalLoss).
            gamma (float): parameter for the classification loss (FocalLoss).
            delta (float): parameter for the localization loss (SmoothL1Loss).
        """
        super(RetinaNetLoss, self).__init__(reduction="auto")
        self._num_classes = num_classes  
        self._cls_loss = FocalLoss(alpha, gamma)
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
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        ### Compute classification loss
        cls_pred = y_pred[:, :, 4:]
        cls_true = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), depth=self._num_classes, dtype=tf.float32)
        cls_loss = self._cls_loss(cls_pred, cls_true)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
        cls_loss = tf.reduce_sum(cls_loss, axis=-1)
        ### Compute localization loss
        box_pred = y_pred[:, :, :4]
        box_true = y_true[:, :, :4]
        box_loss = self._box_loss(box_pred, box_true)
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        box_loss = tf.reduce_sum(box_loss, axis=-1)
        ### Normalize losses
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(cls_loss, normalizer)
        box_loss = tf.math.divide_no_nan(box_loss, normalizer)
        ### Compute total loss
        total_loss = cls_loss + box_loss
        return total_loss
