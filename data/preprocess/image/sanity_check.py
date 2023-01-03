import tensorflow as tf


def sanity_check_classification(image, label):
    """This function asserts the following:
    * image is a 3-D tensorflow tensor of shape (H, W, C) and dtype tf.float32.
    * label is a 0-D tensorflow tensor of dtype tf.int64.
    """
    # check image
    assert len(image.shape) == 3, f"{image.shape=}"
    assert image.dtype == tf.float32, f"{image.dtype=}"
    # check label
    assert label.dtype == tf.int64, f"{label.dtype=}"


def sanity_check_segmentation(image, label):
    """This function asserts the following:
    * image is a 3-D tensorflow tensor of shape (H, W, C) and dtype tf.float32.
    * label is a 2-D tensorflow tensor of shape (H, W) and dtype tf.int64.
    """
    # check image
    assert len(image.shape) == 3, f"{image.shape=}"
    assert image.dtype == tf.float32, f"{image.dtype=}"
    # check label
    assert label.dtype == tf.int64, f"{label.dtype=}"


def sanity_check(image, label):
    assert isinstance(image, tf.Tensor) and isinstance(label, tf.Tensor)
    if len(label.shape) == 0:
        # for classification task
        sanity_check_classification(image, label)
    elif len(label.shape) == 2 and label.shape == image.shape[:2]:
        # for segmentation task
        sanity_check_segmentation(image, label)
    else:
        raise ValueError(f"[ERROR] Argument \'label\' shape out of control. Got {label.shape}.")
