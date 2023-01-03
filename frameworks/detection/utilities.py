import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def swap_xy_coord(boxes):
    assert len(boxes.shape) == 2
    assert boxes.shape[-1] == 4
    return tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis=-1)


def corner_to_center(boxes):
    assert boxes.shape[-1] == 4
    return tf.concat([(boxes[...,:2]+boxes[...,2:])/2.0, boxes[...,2:]-boxes[...,:2]], axis=-1)


def center_to_corner(boxes):
    assert boxes.shape[-1] == 4
    return tf.concat([boxes[...,:2]-boxes[...,2:]/2.0, boxes[...,:2]+boxes[...,2:]/2.0], axis=-1)


def get_pairwise_iou(boxes1, boxes2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[1] == 4
    assert len(boxes2.shape) == 2
    assert boxes2.shape[1] == 4
    corners1 = center_to_corner(boxes1)
    corners2 = center_to_corner(boxes2)
    lu = tf.maximum(corners1[:, None, :2], corners2[None, :, :2])
    rd = tf.maximum(corners1[:, None, 2:], corners2[None, :, 2:])
    intersections = tf.maximum(0.0, rd - lu)
    intersections_area = intersections[:, :, 0] * intersections[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    unions_area = boxes1_area[:, None] + boxes2_area[None, :] - intersections_area
    return tf.clip_by_value(intersections_area / tf.maximum(unions_area, 1e-8), 0.0, 1.0)


def visualize_detections(image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax
