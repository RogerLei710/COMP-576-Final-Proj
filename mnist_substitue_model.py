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

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers
import data_augmentor as da
import numpy as np
from util import split_train_test

target_model = keras.models.load_model('saved_models/mnist_target_model')
(_, _), (x_data, y_data) = mnist.load_data()
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

# first 1,000 test labels
# [85, 126, 116, 107, 110, 87, 87, 99, 89, 94]
x_data = x_data.astype("float32") / 255.0
# 1. initial data collection
x_train, x_test, y_train, y_test = split_train_test(x_data, y_data, test_ratio=0.2)

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
def train_sub(model, x_train, y_train, x_test, y_test, epochs, lamda):
	for iter in range(epochs):
		print("Train substitute network round {} / {} ...".format(iter, epochs))
		# train the ith dataset 10 epochs
		model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
		# data augmentation
		# batch_jacobian shape: (batch_size, num_classes, img_rows, img_cols, channels)
		batch_jacobian = da.Jacobian(model, x_train)
		# build indices list to get elements (batch_size, img_rows, img_cols, channels) from batch_jacobian
		indices = []
		for idx in range(batch_jacobian.shape[0]):
			indices.append([idx, y_train[idx]])
		# now we get the elements
		x_gradient = tf.gather_nd(batch_jacobian, indices)
		# x + lambda * sgn(JF(x)[O(x)])
		x_delta = lamda * tf.sign(x_gradient)
		x_new_train = x_train + x_delta
        # Clip data, each pixel is only valid in [0.0, 1.0]
		x_new_train = np.clip(x_new_train, 0.0, 1.0)
		y_new_train_prob = target_model.predict(x_new_train)
		y_new_train = y_new_train_prob.argmax(axis=-1)
		x_train = tf.concat([x_train, x_new_train], 0)
		y_train = tf.concat([y_train, y_new_train], 0)
	# the augmented data from the last time in the loop needs to be trained
	print("Train substitute network round {} / {} ...".format(epochs, epochs))
	model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))


model = create_model()
model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)

train_sub(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=4, lamda=0.001)
print("(substitute model) Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
model.save("saved_models/mnist_substitute_model/")
