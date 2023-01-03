import tensorflow as tf


class YOLOv1Loss(tf.losses.Loss):

    def __init__(self, num_grid_sqrt=7, num_boxes=2, num_classes=20,
        lambda_coord=5, lambda_noobj=0.5):
        self._S = num_grid_sqrt
        self._B = num_boxes
        self._C = num_classes
        self._lambda_coord = lambda_coord
        self._lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: 3-D tensor of shape (batch_size, 
            y_pred: 3-D tensor of shape (batch_size, S, S, B * 5 + C).
        Returns:
            total_loss: 1-D tensor of shape (batch_size,).
        """
        assert len(y_pred.shape) == 4
        assert y_pred.shape[1] == self._S and y_pred.shape[2] == self._S
        assert y_pred.shape[3] == self._B * 5 + self._C
