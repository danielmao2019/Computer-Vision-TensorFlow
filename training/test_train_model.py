import pytest
from training import train_model
import tensorflow as tf
import data
import models


@pytest.mark.slow
@pytest.mark.parametrize("image_size", [
    (128, 128),
])
def test_train_model(image_size):
    dataset = data.datasets.MNISTDataset(purpose='training').take(16)
    preprocessor = data.preprocess.Preprocessor(transforms=[
        data.preprocess.image.Resize(size=image_size),
    ])
    dataloader = data.dataloaders.Dataloader(dataset=dataset, shuffle=True, preprocessor=preprocessor, batch_size=4)
    print(f"{len(dataloader)=}")
    model = models.cnn.AlexNet(num_classes=10)
    model = model.build(input_shape=image_size+(1,))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.SGD(learning_rate=1.0e-03)
    train_model(specs={
        'tag': 'pipeline_testing',
        'dataloader': dataloader,
        'model': model,
        'loss': loss,
        'optimizer': optimizer,
        'metric': lambda x, y: 0,
        'epochs': 1,
        'save_model': False,
        'load_model': None,
    })
