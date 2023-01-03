import tensorflow as tf


class DecoderYOLOv1:

    def __init__(self, num_grid_sqrt, num_boxes, num_classes,
                 max_detections_per_class=100, max_detections=100,
                 nms_iou_threshold=0.5, confidence_threshold=0.005,
                 box_variance=[0.1, 0.1, 0.2, 0.2], **kwargs):
        self._S = num_grid_sqrt
        self._B = num_boxes
        self._C = num_classes
        self._max_detections_per_class = max_detections_per_class
        self._max_detections = max_detections
        self._nms_iou_threshold = nms_iou_threshold
        self._confidence_threshold = confidence_threshold

    def decode(self, images, predictions):
        """
        Arguments:
            images: 4-D tensor of shape (batch_size, image_height, image_width, image_depth).
            predictions: 4-D tensor of shape (batch_size, S, S, B * 5 + C).
        Returns:
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections.
        """
        batch_size = tf.shape(images)[0]
        assert predictions.shape[0] == batch_size
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        boxes = predictions[:, :, :, :4]
        boxes = tf.reshape(tensor=boxes,
            shape=(batch_size, self._S * self._S, 4))
        scores = predictions[:, :, :, 4] * predictions[:, :, :, 5:]
        scores = tf.reshape(tensor=scores,
            shape=(batch_size, self._S * self._S, self._C))
        return tf.image.combined_non_max_suppression(
            boxes=tf.expand_dims(boxes, axis=2),
            scores=scores,
            max_output_size_per_class=self._max_detections_per_class,
            max_total_size=self._max_detections,
            iou_threshold=self._nms_iou_threshold,
            score_threshold=self._confidence_threshold,
            clip_boxes=False,
        )
