import tensorflow as tf


class SmoothL1Loss(tf.losses.Loss):
    """Localization loss function for 2D object detection."""

    def __init__(self, delta):
        """
        Arguments:
            delta (float): threshold for using absolute loss or squared loss.
        """
        super(SmoothL1Loss, self).__init__(reduction="none")
        self._delta = delta

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: 3-D tensor of shape (batch_size, num_pred, 4).
            y_pred: 3-D tensor of shape (batch_size, num_pred, 4).
        Returns:
            reduced_loss: 2-D tensor of shape (batch_size, num_pred).
        """
        diff = y_true - y_pred
        absolute = tf.abs(diff)
        squared = diff ** 2
        loss = tf.where(tf.less(absolute, self._delta), 0.5 * squared, absolute - 0.5)
        reduced_loss = tf.reduce_sum(loss, axis=-1)
        return reduced_loss
