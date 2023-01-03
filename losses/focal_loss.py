import tensorflow as tf


class FocalLoss(tf.losses.Loss):
    """Focal loss function for 2D object detection."""

    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__(reduction="none")
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: 3-D tensor of shape (batch_size, num_pred, num_classes).
            y_pred: 3-D tensor of shape (batch_size, num_pred, num_classes).
        Returns:
            reduced_loss: 2-D tensor of shape (batch_size, num_pred).
        """
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        alpha = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0-self.alpha)
        probs = tf.nn.sigmoid(y_pred)
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1.0-probs)
        loss = alpha * tf.pow(1.0-pt, self.gamma) * cross_entropy
        reduced_loss = tf.reduce_sum(loss, axis=-1)
        return reduced_loss
