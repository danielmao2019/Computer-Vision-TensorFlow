from train_model import train_model
import tensorflow as tf
import data
import models


if __name__ == "__main__":
    image_h = image_w = 128
    dataset = data.datasets.MNISTDataset(purpose='training').take(16)
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.Resize(size=(image_h, image_w)),
    ])
    dataloader = data.dataloaders.Dataloader(dataset=dataset, shuffle=True, preprocessor=preprocessor, batch_size=4)
    print(f"{len(dataloader)=}")
    model = models.cnn.AlexNet(num_classes=10)
    model = model.build(input_shape=(image_h, image_w, 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.SGD(learning_rate=1.0e-03)
    train_model(specs={
        'tag': 'pipeline_testing',
        'dataloader': dataloader,
        'model': model,
        'loss': loss,
        'optimizer': optimizer,
        'metric': lambda x, y: 0,
        'epochs': 10,
        'save_model': False,
        'load_model': None,
    })
