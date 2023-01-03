import os
import tensorflow as tf

from frameworks.detection.utilities import *
from frameworks.detection.anchor_boxes_generator import *
from frameworks.detection.preprocessor import *
from frameworks.detection.encoder import *
from frameworks.detection.dataloader import *
from frameworks.detection.decoder import *

from losses.RetinaNet_loss import RetinaNetLoss

from models.detection.RetinaNet import get_backbone, RetinaNet


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

MODEL_DIR = "retinanet/"

DATA_DIR = "datasets"  # using downloaded weights

num_classes = 80
BATCH_SIZE = 1
IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512

print("Initialize and Compile Model")
loss = RetinaNetLoss(num_classes)
learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries, values=learning_rates)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model = RetinaNet(num_classes, get_backbone())
model.compile(loss=loss, optimizer=optimizer)

print("Train Model")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=True,
    mode='auto', baseline=None, restore_best_weights=False
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "weights" + "_epoch_{epoch}"),
    monitor="loss",
    save_best_only=False,
    save_weights_only=True,
    verbose=1,
)

callbacks_list = [early_stopping, model_checkpoint]

train_dataset, valid_dataset, _ = DataLoader(dataset_dir=DATA_DIR,
    image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, batch_size=BATCH_SIZE,
).get_train_valid_datasets()
image = next(iter(train_dataset))[0]
print(image.shape)

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=50,
          callbacks=callbacks_list,
          verbose=True,
)

# latest_checkpoint = tf.train.latest_checkpoint(DATADIR)
# model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = Decoder(confidence_threshold=0.5).decode(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

def prepare_image(image):
    image, _, ratio = Preprocessor().resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


val_dataset, dataset_info = tfds.load("coco/2017", split="validation", data_dir=DATA_DIR, with_info=True)
int2str = dataset_info.features["objects"]["label"].int2str

print("inference")

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
