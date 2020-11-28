################################################################################
# Substitute CNN model trained with the help of a target model on Mnist dataset using tf2.
#
# It saves/overwrites model and its configs under saved_models/mnist_substitute_model/.
#
# Usage: python3 mnist_substitue_model.py
#
# Author: Roger
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers

target_model = keras.models.load_model('saved_models/mnist_target_model')
(_, _), (x_data, y_data) = mnist.load_data()
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

# first 1,000 test labels
# [85, 126, 116, 107, 110, 87, 87, 99, 89, 94]
x_data = x_data.astype("float32") / 255.0
# 1. initial data collection
x_train = x_data[:1000]
x_test = x_data[1000:]
y_test = y_data[1000:]

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train_prob = target_model.predict(x_train)
y_train = y_train_prob.argmax(axis=-1)


# 2. substitute model architecture
def create_model():
	model = keras.Sequential(
		[
			keras.Input(shape=(28, 28, 1)),
			layers.Conv2D(16, kernel_size=(5, 5), strides=(3, 3), padding='same', activation='relu'),
			layers.AveragePooling2D(pool_size=(2, 2)),
			layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'),
			layers.AveragePooling2D(pool_size=(2, 2)),
			layers.Flatten(),
			layers.Dense(120, activation='relu'),
			layers.Dense(84, activation='relu'),
			layers.Dense(num_classes, activation='softmax')
		]
	)
	return model


# 3. substitute training
def train_sub(model, epochs):
	for i in range(epochs):
		model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))

	pass


model = create_model()
model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)

# model.fit(x_train, y_train, batch_size=256, epochs=20, validation_data=(x_test, y_test))
# print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
# model.save("saved_models/mnist_substitute_model/")
