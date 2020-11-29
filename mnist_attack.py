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

def get_substitute_model(loc='saved_models/mnist_substitute_model/576final'):
	print('Get substitute model from {}'.format(loc))
	substitute_model = keras.models.load_model(loc)
	substitute_model.compile(
		loss=keras.losses.SparseCategoricalCrossentropy(),
		optimizer=keras.optimizers.Adam(lr=1e-3),
		metrics=["accuracy"],
	)
	return substitute_model

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


def generate_adversarials(batch_size, epslion=0.2):
	while True:
		x = []
		y = []
		for batch in range(batch_size):
			# N = random.randint(0, y_test.shape[0] - 1)
			image = x_test[batch]
			label = y_test_one_hot[batch]

			adversarial = make_adversarial(image, label, epslion)

			x.append(adversarial)
			y.append(y_test[batch])

		x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
		y = np.asarray(y)

		yield x, y


# plot first x misclassification pictures
def plot_misclassifications(miscount, idx, epslion=0.2):
	while miscount > 0:
		image = x_test[idx]
		image = image.reshape((1, img_rows, img_cols, channels))
		image_label = y_test_one_hot[idx]
		adversarial = make_adversarial(image, image_label, epslion)
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


# Run experiment with target model and substitute model (varies)
def run_experiment(substitute_model_loc='saved_models/mnist_substitute_model/576final', plot=False):
	global substitute_model
	substitute_model = get_substitute_model(loc=substitute_model_loc)
	adversarial_num = 1000
	x_adversarial_test, y_adversarial_test = next(generate_adversarials(batch_size=adversarial_num, epslion=0.2))
	print("Accuracy of Substitute model on regular images:", substitute_model.evaluate(x=x_test, y=y_test, verbose=0))
	print("Accuracy of Target model on regular images:", target_model.evaluate(x=x_test, y=y_test, verbose=0))

	print("Accuracy of Substitute model on adversarial images:", substitute_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
	print("Accuracy of Target model on adversarial images:", target_model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

	if plot:
		plot_misclassifications(miscount=10, idx=0, epslion=0.2)


if __name__ == "__main__":
	run_experiment()
