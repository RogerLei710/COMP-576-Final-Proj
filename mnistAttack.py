################################################################################
# Demo script that immplements white box attack on trained CNN on Mnist dataset.
#
# It loads stored model from saved_models/mnist, using its gradient directly to
# generate a perturbed image for each input.
#
# Usage: python3 mnistAttack.py
# 
# Author: Roger
################################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

model = keras.models.load_model('saved_models/mnist')
model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)


def adversarial_pattern(image, label):
	image = tf.cast(image, tf.float32)
	with tf.GradientTape() as tape:
		tape.watch(image)
		prediction = model(image)
		loss = tf.keras.losses.MSE(label, prediction)
	gradient = tape.gradient(loss, image)
	signed_grad = tf.sign(gradient)
	return signed_grad


def generate_adversarials(batch_size):
	while True:
		x = []
		y = []
		for batch in range(batch_size):
			N = random.randint(0, y_test.shape[0])
			image = x_test[N]
			label = y_test_one_hot[N]

			perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()

			epsilon = 0.1
			adversarial = image + perturbations * epsilon

			x.append(adversarial)
			y.append(y_test[N])

		x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
		y = np.asarray(y)

		yield x, y


# plot first x misclassification pictures
def plot_misclassifications(miscount, idx):
	while miscount > 0:
		image = x_test[idx]
		image = image.reshape((1, img_rows, img_cols, channels))
		image_label = y_test_one_hot[idx]
		perturbations = adversarial_pattern(image, image_label).numpy()
		adversarial = image + perturbations * 0.1
		if labels[model.predict(image).argmax()] != labels[model.predict(adversarial).argmax()]:
			miscount -= 1
			if channels == 1:
				plt.imshow(adversarial.reshape((img_rows, img_cols)))
			else:
				plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
			plt.show()
			print('The original image is classified as {}'.format(labels[model.predict(image).argmax()]))
			print('The adversarial image is classified as {}'.format(labels[model.predict(adversarial).argmax()]))
		idx += 1


adversarial_num = 1000
x_adversarial_test, y_adversarial_test = next(generate_adversarials(adversarial_num))
# print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
# print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

plot_misclassifications(10, 0)
