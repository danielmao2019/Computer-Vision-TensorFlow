from ResNet import ResNet
import tensorflow as tf
import data


if __name__ == "__main__":
    image_h = image_w = 224
    dataset = data.datasets.MNIST.MNISTDataset(purpose='training').tensorflow()
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.ToTensor(),
        data.preprocess.image.Resize(size=(image_h, image_w)),
    ])
    dataset = dataset.map(preprocessor)
    model = ResNet(num_classes=dataset.NUM_CLASSES, version=18)
    model = model.build(input_shape=(image_h, image_w, 1))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model.fit(dataset)
