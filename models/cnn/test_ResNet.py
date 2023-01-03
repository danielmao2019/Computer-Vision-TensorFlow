import tensorflow as tf
from ResNet import ResNet

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

train_size = X_train.shape[0]
test_size = X_test.shape[0]

X_train = X_train.reshape(train_size, 224, 224, 1)
X_test = X_test.reshape(test_size, 224, 224, 1)
y_train = y_train.reshape(train_size, 1)
y_test = y_test.reshape(test_size, 1)
X_train = X_train / 255.
X_test = X_test / 255.

print(X_train.shape)
print(y_train.shape)

model = ResNet(num_classes=10, version=18)
model = model.build(input_shape=(224, 224, 1))
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(x=X_train, y=y_train, batch_size=16)
