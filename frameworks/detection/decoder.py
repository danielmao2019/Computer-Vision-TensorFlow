import tensorflow as tf
from frameworks.detection.utilities import center_to_corner
from frameworks.detection.anchor_boxes_generator import *


class Decoder:

    def __init__(self, num_classes=80,
                 max_detections_per_class=100, max_detections=100,
                 nms_iou_threshold=0.5, confidence_threshold=0.005,
                 box_variance=[0.1, 0.1, 0.2, 0.2], **kwargs):
        self._num_classes = num_classes
        self._max_detections_per_class = max_detections_per_class
        self._max_detections = max_detections
        self._nms_iou_threshold = nms_iou_threshold
        self._confidence_threshold = confidence_threshold
        self._box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)
        self._anchor_boxes_generator = AnchorBoxesGenerator()

    def _decode_box_predictions(self, anchor_boxes, box_pred):
        """
        Arguments:
            anchor_boxes: 3-D tensor of shape (batch_size, num_anchor_boxes, 4).
            box_pred: 3-D tensor of shape (batch_size, num_anchor_boxes, 4).
        Returns:
            3-D tensor of shape (batch_size, num_anchor_boxes, 4).
        """
        boxes = box_pred * self._box_variance
        boxes = tf.concat([
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ], axis=-1)
        boxes = center_to_corner(boxes)
        return boxes

    def decode(self, images, predictions):
        """
        Arguments:
            images: 4-D tensor of shape (batch_size, image_height, image_width, image_depth).
            predictions: 3-D tensor of shape (batch_size, num_anchor_boxes, 4 + num_classes).
        Returns:
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections.
        """
        assert len(images.shape) == 4
        assert len(predictions.shape) == 3 and predictions.shape[2] == 4 + self._num_classes
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_boxes_generator.get_anchor_boxes(image_shape[1], image_shape[2])
        box_pred = predictions[:, :, :4]
        boxes = self._decode_box_predictions(anchor_boxes[None, :, :], box_pred)
        cls_pred = predictions[:, :, 4:]
        scores = tf.nn.sigmoid(cls_pred)
        return tf.image.combined_non_max_suppression(
            boxes=tf.expand_dims(boxes, axis=2),
            scores=scores,
            max_output_size_per_class=self._max_detections_per_class,
            max_total_size=self._max_detections,
            iou_threshold=self._nms_iou_threshold,
            score_threshold=self._confidence_threshold,
            clip_boxes=False,
        )
