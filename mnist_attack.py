################################################################################
# Demo script that immplements white box attack on trained CNN on Mnist dataset.
#
# It loads stored model from saved_models/mnist, using its gradient directly to
# generate a perturbed image for each input.
#
# Usage: python3 mnist_attack.py
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


(_, _), (x_test, y_test) = mnist.load_data()
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_test = x_test.astype("float32") / 255.0

x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

substitute_model = keras.models.load_model('saved_models/mnist_substitute_model')
substitute_model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)
target_model = keras.models.load_model('saved_models/mnist_target_model')
target_model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"],
)

def make_adversarial(image, label, epsilon=0.2):
	perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()
	adversarial = np.clip(image + perturbations * epsilon, 0.0, 1.0)
	return adversarial

def adversarial_pattern(image, label):
	image = tf.cast(image, tf.float32)
	with tf.GradientTape() as tape:
		tape.watch(image)
		prediction = substitute_model(image)
		loss = tf.keras.losses.MSE(label, prediction)
	gradient = tape.gradient(loss, image)
	signed_grad = tf.sign(gradient)
	return signed_grad


def generate_adversarials(batch_size):
	while True:
		x = []
		y = []
		for batch in range(batch_size):
			N = random.randint(0, y_test.shape[0] - 1)
			image = x_test[N]
			label = y_test_one_hot[N]

			adversarial = make_adversarial(image, label)

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
		adversarial = make_adversarial(image, image_label)
		if labels[substitute_model.predict(image).argmax()] != labels[substitute_model.predict(adversarial).argmax()]:
			miscount -= 1
			label_orig, label_classified = labels[substitute_model.predict(image).argmax()], labels[substitute_model.predict(adversarial).argmax()]
			if channels == 1:
				adversarial_img = adversarial.reshape((img_rows, img_cols))
				plt.imshow(adversarial_img)
			else:
				plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
			plt.show()
			print('The original image is classified as {}'.format(label_orig))
			print('The adversarial image is classified as {}'.format(label_classified))
		idx += 1


adversarial_num = 1000
x_adversarial_test, y_adversarial_test = next(generate_adversarials(adversarial_num))
print("Accuracy of Substitute model on regular images:", substitute_model.evaluate(x=x_test, y=y_test, verbose=0))
print("Accuracy of target model on regular images:", target_model.evaluate(x=x_test, y=y_test, verbose=0))

print("Accuracy of Substitute model on adversarial images:", substitute_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
print("Accuracy of target model on adversarial images:", target_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

# plot_misclassifications(10, 0)
