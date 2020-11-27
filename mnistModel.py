################################################################################
# Baseline CNN model on Mnist dataset using tf2.
#
# It saves/overwrites model and its configs under saved_models/mnist/.
#
# Usage: python3 mnistModel.py
#
# Author: Roger
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))


def create_model():
	model = keras.Sequential(
		[
			keras.Input(shape=(28, 28, 1)),
			layers.Conv2D(32, kernel_size=(5, 5), strides=(3, 3), padding='same', activation='relu'),
			layers.Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
			layers.Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
			layers.MaxPooling2D(pool_size=(2, 2)),
			layers.Dropout(0.2),
			layers.Flatten(),
			layers.Dense(32, activation='relu'),
			layers.Dropout(0.2),
			layers.Dense(num_classes, activation='softmax')
		]
	)
	return model


model = create_model()
model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
model.save("saved_models/mnist/")
