import tensorflow as tf
from frameworks.detection.utilities import *
from frameworks.detection.anchor_boxes_generator import *


class Encoder:

    def __init__(self, anchor_boxes_generator=None, box_variance=[0.1, 0.1, 0.2, 0.2]):
        self.anchor_boxes_generator = anchor_boxes_generator if anchor_boxes_generator else AnchorBoxesGenerator()
        self.box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

    def match_anchor_boxes(self, anchor_boxes, gt_boxes, match_threshold=0.5, ignore_threshold=0.4):
        """
        Arguments:
            anchor_boxes: 2-D tensor of shape (num_anchor_boxes, 4)
            gt_boxes: 2-D tensor of shape (num_objects, 4)
            match_threshold: anchor_boxes with max iou with the gt_boxes >= match_threshold will be matched
            ignore_threshold: anchor_boxes with max iou with the gt_boxes in the interval [ignore_threshold, match_threshold] will be ignored

        Returns:
            matched_gt_box_idx: 1-D tensor of shape (num_anchor_boxes,)
            positive_mask: 1-D tensor of shape (num_anchor_boxes,)
            ignore_mask: 1-D tensor of shape (num_anchor_boxes,)
        """
        assert len(anchor_boxes.shape) == 2 and anchor_boxes.shape[1] == 4
        assert len(gt_boxes.shape) == 2 and gt_boxes.shape[1] == 4
        pairwise_iou = get_pairwise_iou(anchor_boxes, gt_boxes)
        matched_gt_box_idx = tf.argmax(pairwise_iou, axis=1)  # 1-D tensor of shape (num_anchor_boxes,)
        row_max = tf.reduce_max(pairwise_iou, axis=1)
        positive_mask = tf.greater_equal(row_max, match_threshold)
        negative_mask = tf.less(row_max, ignore_threshold)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return matched_gt_box_idx, tf.cast(positive_mask, dtype=tf.float32), tf.cast(ignore_mask, dtype=tf.float32)

    def get_box_targets(self, anchor_boxes, matched_gt_boxes):
        """
        Arguments:
            anchor_boxes: 2-D tensor of shape (num_anchor_boxes, 4)
            matched_gt_boxes: 2-D tensor of shape (num_anchor_boxes, 4)
        
        Return:
            box_targets: 2-D tensor of shape (num_anchor_boxes, 4)
        """
        num_anchor_boxes = anchor_boxes.shape[0]
        assert len(anchor_boxes.shape) == 2 and anchor_boxes.shape[1] == 4
        assert len(matched_gt_boxes.shape) == 2 and matched_gt_boxes.shape[0] == num_anchor_boxes and matched_gt_boxes.shape[1] == 4
        box_targets = tf.concat([
            (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
            tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
        ], axis=-1)
        box_targets /= self.box_variance
        assert len(box_targets.shape) == 2 and box_targets.shape[0] == num_anchor_boxes and box_targets.shape[1] == 4
        return box_targets

    def encode_single(self, image_shape, gt_boxes, class_ids):
        """
        Create bounding box and class targets for a single example based on ground truth boxes and class ids.

        Arguments:
            gt_boxes: 2-D tensor of shape (num_objects, 4)
            class_ids: 1-D tensor of shape (num_objects,)

        Return:
            labels: 2-D tensor of shape (num_anchor_boxes, 5)
        """
        assert len(gt_boxes.shape) == 2 and gt_boxes.shape[1] == 4
        assert len(class_ids.shape) == 1 and class_ids.shape[0] == gt_boxes.shape[0]
        ### Create bounding box targets
        anchor_boxes = self.anchor_boxes_generator.get_anchor_boxes(image_height=image_shape[1], image_width=image_shape[2])
        matched_gt_box_idx, positive_mask, ignore_mask = self.match_anchor_boxes(anchor_boxes, gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_box_idx)
        box_targets = self.get_box_targets(anchor_boxes, matched_gt_boxes)
        ### Create class targets
        class_ids = tf.cast(class_ids, dtype=tf.float32)
        cls_targets = tf.gather(class_ids, matched_gt_box_idx)
        cls_targets = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, cls_targets)
        cls_targets = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_targets)
        cls_targets = tf.expand_dims(cls_targets, axis=-1)
        ### Create labels
        labels = tf.concat([box_targets, cls_targets], axis=-1)
        assert len(labels.shape) == 2 and labels.shape[1] == 5
        return labels

    def encode(self, images, gt_boxes, class_ids):
        """
        Create bounding box and class targets for a batch of samples based on ground truth boxes and class ids.
        This function does nothing to the images.

        Arguments:
            images: 4-D tensor of shape (batch_size, image_height, image_width, image_depth)
            gt_boxes: 3-D tensor of shape (batch_size, num_objects, 4)
            class_ids: 2-D tensor of shape (batch_size, num_objects)

        Return:
            labels: 3-D tensor of shape (batch_size, num_anchor_boxes, 5)
        """
        ### check input shapes
        assert len(images.shape) == 4
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[0] == images.shape[0] and gt_boxes.shape[2] == 4
        assert len(class_ids.shape) == 2 and class_ids.shape[0] == images.shape[0] and class_ids.shape[1] == gt_boxes.shape[1]
        ### prepare labels
        labels = tf.TensorArray(dtype=tf.float32, size=tf.shape(images)[0], dynamic_size=True)
        for i in range(tf.shape(images)[0]):
            label_i = self.encode_single(tf.shape(images), gt_boxes[i], class_ids[i])
            labels = labels.write(i, label_i)
        labels = labels.stack()
        ### check output shapes
        assert len(labels.shape) == 3 and labels.shape[2] == 5
        return images, labels
