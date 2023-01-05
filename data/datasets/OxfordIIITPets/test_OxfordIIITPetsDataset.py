from OxfordIIITPetsDataset import OxfordIIITPetsDataset
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == "__main__":
    dataset = OxfordIIITPetsDataset(purpose='training', task='seg')
    example = dataset.get_example()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(tf.cast(example[0], dtype=tf.int64))
    ax1.set_title('Image')
    ax2.imshow(example[1])
    ax2.set_title('Segmentation Mask')
    plt.show()
